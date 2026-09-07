#!/usr/bin/env python
"""센터 결과 내보내기 (실행 단위: outputs/<exp>/<run>/) — 클라이언트 run 폴더 **전체**를 익명화해 복사.

  python scripts/export_results.py --exp exp4c --site A --run run_20260903_101500
포함: 모든 로그(run_center/single/client/errors), fold별 분할, DONE 마커, 학습 이력, 라운드 기록, test 지표 CSV, 예측 npz,
      환자 카탈로그, 단일센터 best.pth·FL global_final.pth·라운드별 global, 설정 yaml, 코드 버전.
익명화: 실제 환자 ID(데이터 폴더명)를 모든 텍스트 파일(.log .json .jsonl .csv .txt)과 파일명에서 Case N으로 치환.
        대응표 _id_map_LOCAL_ONLY.json 은 client_ 폴더에만 남기고 export에서 제외.
"""
import os, sys, json, glob, shutil, argparse, datetime, platform, re, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd, torch

TEXT_EXT = {".log", ".json", ".jsonl", ".csv", ".txt", ".yaml", ".out"}
SKIP_FILES = {"_id_map_LOCAL_ONLY.json"}
SKIP_WEIGHTS = re.compile(r"(r\d+_last\.pth|last\.pth|global_r\d+\.pth)$")   # 재개용 옵티마 상태·라운드별 글로벌(서버에 동일본 존재)은 제외; best/global_final 포함


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--exp", required=True); ap.add_argument("--site", required=True); ap.add_argument("--out", default="outputs"); ap.add_argument("--run", required=True)
    a = ap.parse_args(); root = os.path.join(a.out, a.exp, a.run, f"client_{a.site}"); ex = os.path.join(a.out, a.exp, a.run, f"export_{a.site}")
    if os.path.exists(ex): shutil.rmtree(ex)
    os.makedirs(ex)
    ids = sorted({os.path.basename(x) for f in glob.glob(f"{root}/fold*_split.json") for v in json.load(open(f)).values() for x in v})
    if not ids and os.path.exists(f"{root}/patient_catalog.csv"): ids = sorted(pd.read_csv(f"{root}/patient_catalog.csv").case.astype(str).tolist())
    idmap = {pid: f"Case {i+1}" for i, pid in enumerate(ids)}; json.dump(idmap, open(os.path.join(root, "_id_map_LOCAL_ONLY.json"), "w"), indent=1)
    pat = re.compile("|".join(re.escape(i) for i in sorted(ids, key=len, reverse=True))) if ids else None
    mask = (lambda s: pat.sub(lambda m: idmap[m.group(0)], s)) if pat else (lambda s: s)
    n_files = 0
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f in SKIP_FILES or SKIP_WEIGHTS.search(f): continue
            src = os.path.join(dp, f); rel = os.path.relpath(src, root)
            rel_masked = os.path.join(*[mask(p).replace(" ", "_") for p in rel.split(os.sep)])
            dst = os.path.join(ex, rel_masked); os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.splitext(f)[1] in TEXT_EXT:
                open(dst, "w", encoding="utf-8").write(mask(open(src, encoding="utf-8", errors="replace").read()))
            else: shutil.copy2(src, dst)
            n_files += 1
    rows = []
    for f in sorted(glob.glob(f"{root}/fold*/*/test_metrics.csv")):
        fold = int(f.split("fold")[-1].split("/")[0]); method = os.path.basename(os.path.dirname(f)); df = pd.read_csv(f)
        df.insert(0, "method", method); df.insert(0, "fold", fold); df["case"] = df["case"].astype(str).map(lambda c: idmap.get(c, "Case ?")); rows.append(df)
    if rows:
        long = pd.concat(rows); long.to_csv(os.path.join(ex, "metrics_long.csv"), index=False)
        seg = long[long.segment != "liver"]; summ = seg.groupby(["method", "fold"]).dice.mean().unstack().round(4)
        summ["mean"] = summ.mean(axis=1).round(4); summ["sd"] = summ.iloc[:, :-1].std(axis=1).round(4); summ.to_csv(os.path.join(ex, "summary_by_method_fold.csv"))
        json.dump({m: {"dice_mean": float(g.dice.mean()), "hd95_mean": float(g.hd95_mm.mean()), "n_rows": int(len(g))} for m, g in seg.groupby("method")},
                  open(os.path.join(ex, "summary.json"), "w"), indent=1)
    cfg = f"configs/{a.exp}.yaml"
    if os.path.exists(cfg): shutil.copy(cfg, os.path.join(ex, f"config_{a.exp}.yaml"))
    try: git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: git = ""
    json.dump({"site": a.site, "exp": a.exp, "run": a.run, "exported": str(datetime.datetime.now()), "host": platform.node(), "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "n_patients": len(ids), "n_files": n_files, "code_version": git},
              open(os.path.join(ex, "run_info.json"), "w"), indent=1)
    print(f"export → {ex} ({n_files} files, patients {len(ids)}, id map kept locally)")


if __name__ == "__main__":
    main()
