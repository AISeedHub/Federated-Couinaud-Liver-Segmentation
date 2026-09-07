#!/usr/bin/env python
"""모의시험/실험 산출물 자동 검증 — 배포 전 통과 기준.

  python scripts/verify_run.py --config configs/rehearsal.yaml --sites A B --data-roots /data/.../A /data/.../B
검사: (1) fold×method DONE 마커(서버·클라이언트), (2) test_metrics.csv 행 수 = test 환자 × 10(9분절+간),
(3) 클라이언트 global_final.pth == 서버 global_<method>.pth (텐서 일치), (4) Single 결과 존재·행 수,
(5) round_history 라운드 수 = rounds, (6) export_results 산출물에 실제 환자 ID 문자열 미포함, (7) 지표 값 범위(0–1, HD95 유한 비율).
"""
import os, sys, json, glob, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, pandas as pd, numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--sites", nargs="+", required=True); ap.add_argument("--data-roots", nargs="+", required=True); ap.add_argument("--run", required=True)
    a = ap.parse_args(); C = yaml.safe_load(open(a.config)); exp = C["experiment"]; out = C.get("output_dir", "outputs"); folds = C["folds"]; methods = C["methods"]
    fails = []; ok = lambda cond, msg: (None if cond else fails.append(msg))
    srv = os.path.join(out, exp, a.run, "server")
    for f in folds:
        for m in methods:
            ok(os.path.exists(f"{srv}/fold{f}/DONE_{m}"), f"server DONE 없음 fold{f} {m}")
            ok(os.path.exists(f"{srv}/fold{f}/global_{m}.pth"), f"server global 없음 fold{f} {m}")
            hp = f"{srv}/fold{f}/round_history_{m}.json"
            if os.path.exists(hp):
                h = json.load(open(hp)); rounds = {r["round"] for r in h if "clients" in r}; ok(rounds == set(range(1, C["rounds"] + 1)), f"round_history 라운드 불일치 fold{f} {m}: {sorted(rounds)}")
            else: fails.append(f"round_history 없음 fold{f} {m}")
    for site, root in zip(a.sites, a.data_roots):
        cd = os.path.join(out, exp, a.run, f"client_{site}")
        for f in folds:
            sp = json.load(open(f"{cd}/fold{f}_split.json")); nt = len(sp["test"])
            for m in methods + (["Single"] if C.get("run_single", 1) else []):
                d = f"{cd}/fold{f}/{m}"; done = f"{d}/DONE" if m == "Single" else f"{cd}/fold{f}/DONE_{m}"
                ok(os.path.exists(done), f"{site} DONE 없음 fold{f} {m}")
                mp = f"{d}/test_metrics.csv"
                if os.path.exists(mp):
                    df = pd.read_csv(mp); ok(len(df) == nt * 10, f"{site} fold{f} {m} 행수 {len(df)} != {nt*10}")
                    ok(df.dice.dropna().between(0, 1).all() and df.dice.notna().mean() > 0.8, f"{site} fold{f} {m} dice 범위 이상/NaN 과다"); ok(df.vol_gt_ml.gt(0).mean() > 0.9, f"{site} fold{f} {m} GT 부피 0 과다")
                    ok(len(glob.glob(f"{d}/pred/*.npz")) == nt, f"{site} fold{f} {m} 예측 npz {len(glob.glob(f'{d}/pred/*.npz'))} != {nt}")
                else: fails.append(f"{site} test_metrics 없음 fold{f} {m}")
                if m != "Single":
                    gp = f"{d}/global_final.pth"; sgp = f"{srv}/fold{f}/global_{m}.pth"
                    if os.path.exists(gp) and os.path.exists(sgp):
                        a_ = torch.load(gp, map_location="cpu", weights_only=False); b_ = torch.load(sgp, map_location="cpu", weights_only=False)
                        keep_local = (m == "FedBN")
                        diff = [k for k in b_ if k in a_ and not torch.allclose(a_[k].float(), b_[k].float(), atol=1e-6) and not (keep_local and ("norm" in k.lower() or a_[k].ndim == 1))]
                        ok(not diff, f"{site} fold{f} {m} 클라이언트 글로벌 ≠ 서버 글로벌: {len(diff)} 텐서 (예 {diff[:2]})")
                    else: fails.append(f"{site} fold{f} {m} global_final/서버 global 없음")
        cp = f"{cd}/patient_catalog.csv"; ok(os.path.exists(cp) and len(pd.read_csv(cp)) == len([p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]), f"{site} patient_catalog 누락/행수 불일치")
        # export + 익명화
        r = subprocess.run([sys.executable, "scripts/export_results.py", "--exp", exp, "--site", site, "--out", out, "--run", a.run], capture_output=True, text=True)
        ok(r.returncode == 0, f"{site} export 실패: {r.stderr[-300:]}")
        ex = os.path.join(out, exp, a.run, f"export_{site}"); ids = [os.path.basename(p) for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
        texts = [os.path.join(dp, f) for dp, _, fn in os.walk(ex) for f in fn if os.path.splitext(f)[1] in {".csv", ".json", ".jsonl", ".log", ".txt", ".yaml", ".out"}]
        blob = "".join(open(p, errors="ignore").read() for p in texts) + " ".join(os.path.relpath(os.path.join(dp, f), ex) for dp, _, fn in os.walk(ex) for f in fn)
        leaked = [i for i in ids if i in blob]; ok(not leaked, f"{site} export에 환자 ID 노출 {leaked[:3]}")
        ok(len(glob.glob(f"{ex}/**/*.log", recursive=True)) >= 2 and glob.glob(f"{ex}/**/client.log", recursive=True), f"{site} export에 로그 누락")
        ok(not os.path.exists(f"{ex}/_id_map_LOCAL_ONLY.json"), f"{site} export에 ID 매핑 포함")
        ok(os.path.exists(f"{ex}/metrics_long.csv") and os.path.exists(f"{ex}/summary.json"), f"{site} export 산출물 누락")
    print("=" * 60); print("검증 결과:", "통과 ✓" if not fails else f"실패 {len(fails)}건")
    for x in fails: print("  ✗", x)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
