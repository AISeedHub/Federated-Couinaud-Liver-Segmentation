"""공용 학습 루프 — 사전학습·단일센터·FL 로컬 에폭이 모두 이 함수를 쓴다.

nnU-Net 관례: SGD(momentum 0.99, nesterov), lr 0.01, poly decay(0.9), deep supervision 가중치 1/2^i.
중단 안전: 출력 폴더의 STOP.txt 감지 시 에폭 경계에서 정상 종료, last.pth로 재개(resume).
"""
from __future__ import annotations
import os, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import CouinaudDataset, load_case
from .losses import seg_loss, dice_per_class, merge_pred_9
from .infer import predict_volume


def ds_weights(n: int):
    w = [1 / 2 ** i for i in range(n)]; s = sum(w); return [x / s for x in w]


def _worker_init(_):
    torch.set_num_threads(1)          # 워커 내 OpenMP 과다 스레드 방지(SIGABRT 원인)
    import cv2; cv2.setNumThreads(0)


def make_loader(cases, mode, batch_size=1, num_workers=4, cache=False, augment=True):
    ds = CouinaudDataset(cases, mode, augment=augment, cache=cache)
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == "train"), num_workers=num_workers, worker_init_fn=_worker_init,
                      pin_memory=True, persistent_workers=False, drop_last=(mode == "train"), multiprocessing_context="spawn" if num_workers > 0 else None)


def train_step(model, batch, opt, scaler, device, amp, log=print):
    x = batch["image"]; y = batch["label"]
    if int(y.min()) < 0 or int(y.max()) >= 10 or not torch.isfinite(x).all():   # 라벨/영상 이상 배치는 기록 후 건너뜀 (CUDA assert 방지)
        log(f"[skip-batch] case {batch.get('case')} label range {int(y.min())}..{int(y.max())} finite={bool(torch.isfinite(x).all())}"); return float("nan")
    x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True); s8 = batch["seg8"].to(device)
    with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
        outs = model(x); outs = outs if isinstance(outs, (list, tuple)) else [outs]
        w = ds_weights(len(outs)); loss = 0
        for wi, o in zip(w, outs):
            yi = y if o.shape[2:] == y.shape[1:] else F.interpolate(y[:, None].float(), size=o.shape[2:], mode="nearest")[:, 0].long()
            loss = loss + wi * seg_loss(o, yi, s8)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
    scaler.step(opt); scaler.update()
    return float(loss.item())


@torch.no_grad()
def validate(model, cases, device, amp=True, max_cases=None) -> dict:
    """전체 볼륨 슬라이딩 윈도우 Dice(분절 평균). 8분절 케이스는 4a/4b 병합 기준."""
    model.eval(); per = []
    for cd in (cases[:max_cases] if max_cases else cases):
        img, lab, meta = load_case(cd); pred, _ = predict_volume(model, img, device, amp=amp)
        p = torch.from_numpy(pred.astype(np.int64)); g = torch.from_numpy(lab.astype(np.int64))
        if int(meta.get("segments", 9)) == 8: p, g = merge_pred_9(p), merge_pred_9(g); d = dice_per_class(p, g, 9)[1:]
        else: d = dice_per_class(p, g, 10)[1:]
        per.append(float(torch.nanmean(d)))
    model.train()
    return {"dice": float(np.mean(per)) if per else float("nan"), "n": len(per)}


def fit(model, train_cases, val_cases, out_dir, device, epochs=200, lr=0.01, amp=True, batch_size=1, num_workers=4,
        val_every=5, val_max=20, resume=True, cache=False, log=print, epoch_offset=0, total_epochs=None, save_prefix=""):
    """epochs만큼 학습. total_epochs(전체 일정)로 poly lr을 계산하므로 FL 로컬 에폭도 전역 일정을 따른다."""
    os.makedirs(out_dir, exist_ok=True); total = total_epochs or (epoch_offset + epochs)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    last = os.path.join(out_dir, f"{save_prefix}last.pth"); best_p = os.path.join(out_dir, f"{save_prefix}best.pth")
    start, best = epoch_offset, -1.0; hist = []
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["epoch"] + 1; best = ck.get("best", -1.0); hist = ck.get("hist", [])
        log(f"[resume] epoch {start} best {best:.4f}")
    loader = make_loader(train_cases, "train", batch_size, num_workers, cache=cache); model.to(device).train()
    for ep in range(start, epoch_offset + epochs):
        if os.path.exists(os.path.join(out_dir, "STOP.txt")): log("[stop] STOP.txt 감지 — 에폭 경계에서 종료"); break
        cur = lr * (1 - ep / total) ** 0.9
        for g in opt.param_groups: g["lr"] = cur
        t0 = time.time(); losses = [train_step(model, b, opt, scaler, device, amp, log) for b in loader]
        rec = {"epoch": ep, "loss": float(np.nanmean(losses)), "skipped": int(np.isnan(losses).sum()), "lr": cur, "sec": time.time() - t0}
        if val_cases and ((ep + 1) % val_every == 0 or ep + 1 == epoch_offset + epochs):
            v = validate(model, val_cases, device, amp, val_max); rec["val_dice"] = v["dice"]
            if v["dice"] > best: best = v["dice"]; torch.save(model.state_dict(), best_p)
        hist.append(rec); log(json.dumps(rec))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best, "hist": hist}, last)
    json.dump(hist, open(os.path.join(out_dir, f"{save_prefix}history.json"), "w"), indent=1)
    return {"best_val_dice": best, "best_path": best_p if os.path.exists(best_p) else None, "last_path": last, "history": hist}
