"""연합학습 — Flower 1.11 NumPyClient / 서버 전략.

방법: FedAvg, FedProx(mu), FedAdam(서버 측 Adam, Reddi 2021), FedBN-IN(InstanceNorm affine 파라미터를 로컬에 유지 — BN 없음을 명시한 변형)
클라이언트 원칙
  * 매 라운드 서버가 보낸 글로벌 파라미터를 `last_global`에 스냅샷 → evaluate/최종 테스트는 항상 글로벌로 수행(v1 버그 재발 방지).
  * 서버 config(method, lr, local_epochs, total_epochs, fold)를 그대로 따른다.
  * 학습 결과·테스트 지표는 로컬 outputs/<exp>/fold<k>/<method>/ 에 저장. 환자 ID는 케이스 폴더명 그대로 두되 배포 산출물에서는 Case N으로 치환.
"""
from __future__ import annotations
import os, json, copy
from collections import OrderedDict
import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, Parameters, Scalar
from flwr.server.strategy import FedAvg
from .train import fit, validate
from .data import load_case
from .infer import predict_volume, predict_volume_auto_orient
from .metrics import case_metrics, summarize

METHODS = ["FedAvg", "FedProx", "FedAdam", "FedBN"]


def get_nd(model): return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_nd(model, nds, keep_keys=()):
    sd = model.state_dict(); new = OrderedDict()
    for (k, old), v in zip(sd.items(), nds):
        new[k] = old if any(k.endswith(s) or s in k for s in keep_keys) and keep_keys else torch.from_numpy(np.asarray(v)).to(old.dtype)
    model.load_state_dict(new)


def bn_keys(model):
    """FedBN-IN: InstanceNorm affine(weight/bias) 키 — 로컬 유지 대상."""
    return [k for k, m in model.named_modules() if isinstance(m, torch.nn.InstanceNorm3d)]


class CouinaudClient(fl.client.NumPyClient):
    def __init__(self, model, train_cases, val_cases, test_cases, device, out_dir, amp=False, workers=4, log=print, cid="client"):
        self.model, self.device, self.out, self.log, self.cid = model, device, out_dir, log, cid
        self.train_cases, self.val_cases, self.test_cases = train_cases, val_cases, test_cases
        self.amp, self.workers = amp, workers; self.last_global = None; self.round = 0
        self.in_keys = bn_keys(model); os.makedirs(out_dir, exist_ok=True)

    def get_parameters(self, config): return get_nd(self.model)

    expected_fold = None; expected_method = None   # client.py가 세션 시작 시 지정(미지정=검증 비활성)

    def _check_session(self, config, method):
        """서버가 보낸 (fold, method)와 로컬 기대 세션 대조 — 어긋나면 학습 없이 즉시 중단(라벨 오염 방지)."""
        if self.expected_fold is None: return
        sf = int(config.get("fold", -1))
        if sf != int(self.expected_fold) or method != self.expected_method:
            raise RuntimeError(f"SESSION_MISMATCH: 서버 fold{sf}/{method} vs 로컬 fold{self.expected_fold}/{self.expected_method} — 서버·클라이언트 run 진도가 어긋났습니다. 양쪽을 새 run으로 정렬하세요.")

    def _apply_global(self, params, method):
        keep = self.in_keys if method == "FedBN" else ()
        set_nd(self.model, params, keep); self.last_global = copy.deepcopy(self.model.state_dict())

    def fit(self, parameters, config):
        method = str(config.get("method", "FedAvg")); rnd = int(config.get("server_round", 1)); self.round = rnd
        self._check_session(config, method)
        le = int(config.get("local_epochs", 10)); total = int(config.get("total_epochs", 200)); lr = float(config.get("lr", 0.01))
        self._apply_global(parameters, method)
        prox_mu = float(config.get("fedprox_mu", 0.0)) if method == "FedProx" else 0.0
        if prox_mu > 0: _install_prox(self.model, prox_mu)
        r = fit(self.model, self.train_cases, None, os.path.join(self.out, method), self.device, epochs=le, lr=lr, amp=self.amp,
                num_workers=self.workers, resume=False, log=self.log, epoch_offset=(rnd - 1) * le, total_epochs=total, save_prefix=f"r{rnd:02d}_")
        if prox_mu > 0: _remove_prox(self.model)
        return get_nd(self.model), len(self.train_cases), {"loss": float(r["history"][-1]["loss"]) if r["history"] else 0.0, "cid": self.cid}

    def evaluate(self, parameters, config):
        method = str(config.get("method", "FedAvg")); rnd = int(config.get("server_round", 0)); final = bool(config.get("final", False))
        self._check_session(config, method)
        self._apply_global(parameters, method)
        v = validate(self.model, self.val_cases, self.device, self.amp)
        os.makedirs(os.path.join(self.out, method), exist_ok=True)   # fit 없이 evaluate가 먼저 온 클라(재접속 직후)에서 폴더 부재 크래시 방지
        torch.save(self.last_global, os.path.join(self.out, method, f"global_r{rnd:02d}.pth"))
        rec = {"round": rnd, "val_dice": v["dice"], "cid": self.cid}
        if final or rnd == int(config.get("num_rounds", -1)):
            rec.update(self.final_test(method))
        self.log(json.dumps(rec))
        with open(os.path.join(self.out, method, "rounds.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
        return 1.0 - v["dice"], len(self.val_cases), rec

    @torch.no_grad()
    def final_test(self, method):
        """서버에서 받은 글로벌 가중치(last_global)로 테스트 — 로컬 학습 상태가 아님을 보장."""
        from .lesion import lesion_overlap_rows
        self.model.load_state_dict(self.last_global); self.model.eval(); rows = []; lrows = []
        for cd in self.test_cases:
            img, lab, meta = load_case(cd); pred, _, _fl = predict_volume_auto_orient(self.model, img, self.device, amp=self.amp)
            sp = meta.get("spacing") or [4.0, 0.7, 0.7]
            rows += case_metrics(pred, lab, [float(sp[0]), float(sp[1]), float(sp[2])], os.path.basename(cd))
            lrows += lesion_overlap_rows(os.path.basename(cd), cd, lab, pred, [float(sp[0]), float(sp[1]), float(sp[2])])
            pdir = os.path.join(self.out, method, "pred"); os.makedirs(pdir, exist_ok=True)
            np.savez_compressed(os.path.join(pdir, f"{os.path.basename(cd)}.npz"), pred=pred)   # 라벨맵 원자료(추후 어떤 지표든 재계산)
        import pandas as pd; pd.DataFrame(rows).to_csv(os.path.join(self.out, method, "test_metrics.csv"), index=False)
        if lrows: pd.DataFrame(lrows).to_csv(os.path.join(self.out, method, "lesion_overlap.csv"), index=False)
        torch.save(self.last_global, os.path.join(self.out, method, "global_final.pth"))
        s = summarize(rows); json.dump(s, open(os.path.join(self.out, method, "test_summary.json"), "w"), indent=1)
        return {"test_" + k: v for k, v in s.items() if not isinstance(v, dict)}


# ── FedProx: 옵티마 스텝 전에 (mu/2)||w-w_g||^2 그라디언트를 더하는 훅 ──
def _install_prox(model, mu):
    model._prox_ref = [p.detach().clone() for p in model.parameters()]; model._prox_mu = mu
    model._prox_handles = [p.register_hook(lambda g, p=p, r=r: g + mu * (p.detach() - r)) for p, r in zip(model.parameters(), model._prox_ref)]

def _remove_prox(model):
    for h in getattr(model, "_prox_handles", []): h.remove()
    model._prox_handles = []; model._prox_ref = None


class CouinaudStrategy(FedAvg):
    """FedAvg 상속. FedAdam은 서버측 Adam으로 집계 델타를 적용. 라운드 기록·글로벌 저장."""

    def __init__(self, method, model_keys, out_dir, num_rounds, local_epochs, lr=0.01, fedprox_mu=0.01, server_lr=1e-3, fold=0, round_offset=0, **kw):
        # round_offset: 세션 무효 후 재개 시 이미 집계된 라운드 수. Flower는 1..남은라운드로 돌지만
        # 클라이언트·저장 파일에는 절대 라운드(offset+r)를 쓴다 → 무효 손실이 세션 전체가 아니라 한 라운드로 줄어든다.
        super().__init__(**kw); self.method, self.keys, self.out, self.num_rounds = method, model_keys, out_dir, num_rounds
        self.round_offset = int(round_offset)
        self.local_epochs, self.lr, self.mu, self.server_lr, self.fold = local_epochs, lr, fedprox_mu, server_lr, fold
        self.history = []; self.last_parameters = None; self.m = None; self.v = None; self.t = 0
        self._init_params = kw.get("initial_parameters")   # 세션 재시작 시에도 항상 깨끗한 사전학습 초기값 사용(Flower 기본은 1회 소비 후 파기)
        os.makedirs(out_dir, exist_ok=True)

    def initialize_parameters(self, client_manager):
        return self._init_params

    def aggregate_fit(self, server_round, results, failures):
        # 부분 집계 금지: 결과가 참여 정원(min_fit_clients)에 못 미치면 집계 자체를 거부하고 세션을 무효화한다.
        # (죽은 연결이 명단에 유령으로 남아 '4명 샘플→3개 결과'로 진행되는 Flower 기본 동작이
        #  다기관 full-participation 설계를 오염시키는 것을 원천 차단 — 2026-09-19 실측)
        # 참여·실패 명세를 항상 남긴다 — 미달 시 실패 주체 추적용 (cid = gRPC peer 주소)
        ar = server_round + self.round_offset
        for cp, fr in results:
            print(f"[participation] round {ar} OK cid={cp.cid} site={fr.metrics.get('cid', fr.metrics.get('site','?')) if fr.metrics else '?'}", flush=True)
        for f in failures:
            if isinstance(f, tuple):
                print(f"[participation] round {ar} FAIL cid={f[0].cid} res={f[1]!r}"[:500], flush=True)
            else:
                print(f"[participation] round {ar} FAIL exc={f!r}"[:500], flush=True)
        if len(results) < self.min_fit_clients:
            self.history.append({"round": ar, "n_clients": len(results), "failures": len(failures), "invalid": True})
            raise RuntimeError(f"SESSION_INVALID: round {ar} results {len(results)} < required {self.min_fit_clients}")
        return self._aggregate_fit_impl(server_round, results, failures)

    def _cfg(self, rnd, final=False):
        return {"method": self.method, "server_round": rnd + self.round_offset, "local_epochs": self.local_epochs, "total_epochs": self.num_rounds * self.local_epochs,
                "lr": self.lr, "fedprox_mu": self.mu, "num_rounds": self.num_rounds, "fold": self.fold, "final": final}

    def configure_fit(self, server_round, parameters, client_manager):
        self.on_fit_config_fn = lambda r: self._cfg(r); return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        self.on_evaluate_config_fn = lambda r: self._cfg(r, final=(r + self.round_offset == self.num_rounds)); return super().configure_evaluate(server_round, parameters, client_manager)

    def _aggregate_fit_impl(self, server_round, results, failures):
        agg, metrics = super().aggregate_fit(server_round, results, failures)
        if agg is None: return agg, metrics
        if self.method == "FedAdam" and self.last_parameters is not None:
            prev = parameters_to_ndarrays(self.last_parameters); new = parameters_to_ndarrays(agg)
            delta = [n - p for n, p in zip(new, prev)]; self.t += 1; b1, b2, eps = 0.9, 0.99, 1e-3
            self.m = [b1 * m + (1 - b1) * d for m, d in zip(self.m or [np.zeros_like(d) for d in delta], delta)]
            self.v = [b2 * v + (1 - b2) * d * d for v, d in zip(self.v or [np.zeros_like(d) for d in delta], delta)]
            upd = [p + self.server_lr * m / (np.sqrt(v) + eps) for p, m, v in zip(prev, self.m, self.v)]
            agg = ndarrays_to_parameters(upd)
        self.last_parameters = agg
        self._save_global(server_round + self.round_offset)
        self.history.append({"round": server_round, "n_clients": len(results), "failures": len(failures),
                             "client_loss": {r.metrics.get("cid", c.cid): r.metrics.get("loss") for c, r in results}})
        return agg, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        rec = {"round": server_round + self.round_offset, "clients": {r.metrics.get("cid", c.cid): {k: v for k, v in r.metrics.items() if k != "cid"} for c, r in results}}
        self.history.append(rec); json.dump(self.history, open(os.path.join(self.out, f"round_history_{self.method}.json"), "w"), indent=1)
        return loss, metrics

    def _save_global(self, rnd):
        nds = parameters_to_ndarrays(self.last_parameters)
        sd = OrderedDict((k, torch.from_numpy(np.asarray(v))) for k, v in zip(self.keys, nds))
        torch.save(sd, os.path.join(self.out, f"global_{self.method}_r{rnd:02d}.pth"))
        if rnd == self.num_rounds: torch.save(sd, os.path.join(self.out, f"global_{self.method}.pth"))
