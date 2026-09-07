#!/usr/bin/env python
"""TotalSegmentator 퀴노 모델(570 CT / 576 MR) zero-shot 평가 — TS 공식 파이프라인(리샘플·간 crop 포함) 그대로 사용.

  python scripts/eval_ts_zero_shot.py --task liver_segments --cases /data/campaign/public_v2/01_msd08_tian --list outputs/splits/pretrain_split.json:test_lee50_tian
  python scripts/eval_ts_zero_shot.py --task liver_segments --cases /home/dspserver/2025/jin/combined_80        # 기관 전원 (9→8 병합)
  python scripts/eval_ts_zero_shot.py --task liver_segments_mr --cases /data/campaign/public_v2/06_ts_mr --mr
입력은 uint8 기관 포맷 → 근사 HU NIfTI 복원(CT). MR은 percentile 정규화된 uint8 그대로(0–255) 넘긴다(TS MR은 z-score 정규화).
TS 라벨(liver_segment_1..8) → 우리 8분절 9클래스(0 bg,1..8). 기관 9분절 GT는 4a/4b→4 병합 후 비교.
"""
import os, sys, json, argparse, shutil, tempfile, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, nibabel as nib, pandas as pd, torch
from couinaudfl.data import load_case, list_cases
from couinaudfl.export_nifti import save_nifti, u8_to_hu, load_nifti_as_inst
from couinaudfl.metrics import case_metrics, summarize

ap = argparse.ArgumentParser(); ap.add_argument("--task", default="liver_segments"); ap.add_argument("--cases", required=True)
ap.add_argument("--list", default=None, help="json:key 로 케이스 부분집합"); ap.add_argument("--mr", action="store_true"); ap.add_argument("--out", default=None)
ap.add_argument("--weights", default="/data/datasets/couinaud_public/checkpoints/ts"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-hd", action="store_true")

def main():
    a = ap.parse_args()
    # TS 가중치 경로: <weights>/nnunet/results/Dataset570_... 형태로 심볼릭 링크 구성
    wroot = os.path.join(a.weights, "_tsroot", "nnunet", "results"); os.makedirs(wroot, exist_ok=True)
    for n in os.listdir(a.weights):
        d = os.path.join(a.weights, n)
        if os.path.isdir(d) and n.isdigit():
            for sub in os.listdir(d):
                if sub.startswith("Dataset") and not os.path.exists(os.path.join(wroot, sub)): os.symlink(os.path.join(d, sub), os.path.join(wroot, sub))
    os.environ["TOTALSEG_WEIGHTS_PATH"] = os.path.join(a.weights, "_tsroot")
    from totalsegmentator.python_api import totalsegmentator

    cases = list_cases(a.cases)
    if a.list:
        jp, key = a.list.split(":"); sel = set(os.path.basename(x) for x in json.load(open(jp))[key]); cases = [c for c in cases if os.path.basename(c) in sel]
    if a.limit: cases = cases[:a.limit]
    out = a.out or f"outputs/zeroshot_{a.task}_{os.path.basename(a.cases.rstrip('/'))}"; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    log(f"TS {a.task} zero-shot on {a.cases}: {len(cases)} cases")
    TS_MAP = {i: i for i in range(1, 9)}   # TS ml 출력에서 liver_segment_k 의 정수값은 class_map 순서(1..8)
    rows = []; tmp = tempfile.mkdtemp(prefix="ts_zs_")
    for i, cd in enumerate(cases):
        name = os.path.basename(cd); pred_p = os.path.join(out, f"pred_{name}.npy")
        img, lab, meta = load_case(cd); sp = meta.get("spacing") or meta.get("orig_spacing") or [4.0, 0.7, 0.7]; spacing = [4.0, float(sp[1]), float(sp[2])]
        if os.path.exists(pred_p): pred = np.load(pred_p)
        else:
            inp = os.path.join(tmp, f"{name}.nii.gz"); outp = os.path.join(tmp, f"{name}_seg.nii.gz")
            save_nifti(img.astype(np.float32) if a.mr else u8_to_hu(img), spacing, inp)
            try:
                totalsegmentator(inp, outp, task=a.task, ml=True, quiet=True, fast=False, nr_thr_resamp=1, nr_thr_saving=1, device="gpu")
                seg = load_nifti_as_inst(outp).astype(np.uint8)
                if seg.shape != img.shape: log(f"  shape mismatch {name}: {seg.shape} vs {img.shape}"); continue
                pred = np.zeros_like(seg)
                for k, v in TS_MAP.items(): pred[seg == k] = v
                # pred는 8분절 9클래스(0..8). 우리 10클래스 규약으로 옮김: 4→4(4a 자리), 5..8→6..9
                pred10 = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9], np.uint8)[pred]; pred = pred10; np.save(pred_p, pred)
            except Exception as e:
                log(f"  FAIL {name}: {type(e).__name__}: {str(e)[:120]}"); continue
            finally:
                for p in (inp, outp):
                    if os.path.exists(p): os.remove(p)
        rows += case_metrics(pred, lab, spacing, name, seg8=True, with_hd=not a.no_hd)
        if (i + 1) % 10 == 0: log(f"  {i+1}/{len(cases)} | 누적 분절 Dice {summarize(rows)['dice_mean_segments']:.4f}")
    shutil.rmtree(tmp, ignore_errors=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False); s = summarize(rows)
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()