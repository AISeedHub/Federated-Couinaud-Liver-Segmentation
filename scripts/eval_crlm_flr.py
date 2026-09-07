#!/usr/bin/env python
"""CRLM 197 — FLR(잔여간) 기준 평가.
GT: 1=liver(절제측), 2=remnant(FLR), 3=hepatic vein, 4=portal vein, 5=tumor.
예측 FLR = 예측 분절 중 'GT remnant와 과반(>50%) 겹치는 분절'의 합집합 (임상: 남길 분절 지정 방식 모사).
지표: 전체 간(1∪2∪5) Dice·부피, FLR Dice·부피오차%·FLR비율(FLR/간) 상관, 채택 분절 목록.
  python scripts/eval_crlm_flr.py --weights outputs/pretrain/best.pth --out outputs/zeroshot/crlm_flr
"""
import os, sys, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, pandas as pd
from couinaudfl.model import build_model
from couinaudfl.data import load_case, list_cases, SEG_NAMES
from couinaudfl.infer import predict_volume
from couinaudfl.metrics import dice as _dice, hd95


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--cases", default="/data/campaign/public_v2/07_crlm")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--amp", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda"); m = build_model().to(dev); sd = torch.load(a.weights, map_location=dev, weights_only=False); m.load_state_dict(sd.get("model", sd)); m.eval()
    cases = list_cases(a.cases); cases = cases[:a.limit] if a.limit else cases; log(f"CRLM FLR: {len(cases)} cases | {a.weights}")
    rows = []
    for i, cd in enumerate(cases):
        img, lab_raw, meta = load_case(cd); name = os.path.basename(cd)
        # load_case는 10채널 argmax를 주지만 CRLM mask.npy ch1..5 = liver/remnant/hv/pv/tumor
        sp = meta["spacing"]; vox = float(np.prod(sp)) / 1000.0
        pp = os.path.join(a.out, f"pred_{name}.npy")
        if os.path.exists(pp): pred = np.load(pp)
        else: pred, _ = predict_volume(m, img, dev, amp=bool(a.amp)); np.save(pp, pred)
        gt_liver = np.isin(lab_raw, [1, 2, 5]); gt_flr = lab_raw == 2
        if gt_flr.sum() == 0 or gt_liver.sum() == 0: log(f"  skip {name}: remnant/간 GT 없음"); continue
        pr_liver = pred > 0
        # 분절별 remnant 과반 판정
        sel = []
        for c in range(1, 10):
            pc = pred == c
            if pc.sum() > 0 and (pc & gt_flr).sum() / pc.sum() > 0.5: sel.append(c)
        pr_flr = np.isin(pred, sel)
        r = {"case": name, "liver_dice": _dice(pr_liver, gt_liver), "liver_vol_gt_ml": float(gt_liver.sum() * vox), "liver_vol_pred_ml": float(pr_liver.sum() * vox),
             "flr_dice": _dice(pr_flr, gt_flr) if sel else 0.0, "flr_vol_gt_ml": float(gt_flr.sum() * vox), "flr_vol_pred_ml": float(pr_flr.sum() * vox),
             "flr_hd95_mm": hd95(pr_flr, gt_flr, sp) if sel else float("nan"),
             "flr_pct_gt": float(gt_flr.sum() / gt_liver.sum() * 100), "flr_pct_pred": float(pr_flr.sum() / max(pr_liver.sum(), 1) * 100),
             "segments_selected": "+".join(SEG_NAMES[c - 1] for c in sel)}
        r["flr_vol_err_pct"] = (r["flr_vol_pred_ml"] - r["flr_vol_gt_ml"]) / r["flr_vol_gt_ml"] * 100
        rows.append(r)
        if (i + 1) % 20 == 0:
            df = pd.DataFrame(rows); log(f"  {i+1}/{len(cases)} | liver {df.liver_dice.mean():.4f} FLR {df.flr_dice.mean():.4f}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "metrics_flr.csv"), index=False)
    s = {"n": len(df), "liver_dice": round(float(df.liver_dice.mean()), 4),
         "flr_dice": round(float(df.flr_dice.mean()), 4), "flr_dice_median": round(float(df.flr_dice.median()), 4),
         "flr_hd95_mm": round(float(df.flr_hd95_mm.mean()), 1),
         "flr_vol_abs_err_pct_mean": round(float(df.flr_vol_err_pct.abs().mean()), 2),
         "flr_pct_corr_pearson": round(float(df.flr_pct_gt.corr(df.flr_pct_pred)), 4),
         "flr_vol_corr_pearson": round(float(df.flr_vol_gt_ml.corr(df.flr_vol_pred_ml)), 4)}
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()
