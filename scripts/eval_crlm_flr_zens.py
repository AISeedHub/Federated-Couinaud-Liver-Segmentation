# -*- coding: utf-8 -*-
"""CRLM FLR — z-위상 앙상블 추론(native 정밀도 회복 시험).
원본 DICOM HU 스택을 z 시작 위상 φ∈{0..K-1}mm 씩 어긋난 4mm 격자로 리샘플해 각각 추론,
10클래스 확률을 원본 z 격자로 선형보간해 평균한 뒤 argmax → 원본 GT로 채점.
  .venv/bin/python scripts/eval_crlm_flr_zens.py --weights outputs/pretrain/best.pth \
      --out outputs/zeroshot_baselines/crlm_flr_zens20 --limit 20 --phases 4
기존 eval_crlm_flr_native.py 의 native_gt/채점 규약 재사용. 전처리 창(WL80/WW225)·80슬라이스 규약 동일.
"""
from __future__ import annotations
import os, sys, glob, json, argparse, datetime, importlib.util
import numpy as np, torch, pandas as pd, pydicom
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from couinaudfl.model import build_model
from couinaudfl.data import list_cases, load_case
from couinaudfl.infer import predict_volume
from couinaudfl.metrics import dice as _dice, hd95

spec = importlib.util.spec_from_file_location("cfn", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_crlm_flr_native.py"))
cfn = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfn)
native_gt = cfn.native_gt
TMP = cfn.TMP; ZMM = 4.0; WL, WW = 80.0, 225.0


def native_hu(pid):
    d = os.path.join(TMP, pid)
    files = glob.glob(f"{d}/ct/**/*.dcm", recursive=True)
    dss = [pydicom.dcmread(f) for f in files]
    dss = sorted([x for x in dss if hasattr(x, "ImagePositionPatient")], key=lambda x: -float(x.ImagePositionPatient[2]))
    vol = np.stack([x.pixel_array.astype(np.float32) * float(x.RescaleSlope) + float(x.RescaleIntercept) for x in dss])
    sz = abs(float(dss[0].ImagePositionPatient[2]) - float(dss[1].ImagePositionPatient[2]))
    sy, sx = map(float, dss[0].PixelSpacing)
    return vol, (sz, sy, sx)


def hu_to_u8(hu):
    lo, hi = WL - WW / 2.0, WL + WW / 2.0
    return np.clip((hu - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default="/data/campaign/public_v2/07_crlm")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1); ap.add_argument("--no-hd", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda"); m = build_model().to(dev)
    sd = torch.load(a.weights, map_location=dev, weights_only=False); m.load_state_dict(sd.get("model", sd)); m.eval()
    cases = [c for c in list_cases(a.cases) if os.path.basename(c) != "CRLM-CT-1075"]
    if a.limit: cases = cases[:a.limit]
    offsets = [k * ZMM / a.phases for k in range(a.phases)]   # 0,1,2,3mm (phases=4)
    log(f"z-위상 앙상블 CRLM FLR: {len(cases)} cases, phases={offsets}")
    rows = []
    for i, cd in enumerate(cases):
        pid = os.path.basename(cd)
        try:
            _, _, meta = load_case(cd); z0 = int(meta.get("z0", 0))
            hu, sp_n = native_hu(pid); lab_n, sp_g = native_gt(pid)
            if hu.shape != lab_n.shape: log(f"  skip {pid}: ct/gt shape 상이"); continue
            if hu.shape[1:] != (512, 512): log(f"  skip {pid}: in-plane {hu.shape[1:]}"); continue
            Dn = hu.shape[0]; sz = sp_n[0]
            u8n = hu_to_u8(hu).astype(np.float32)
            acc = None
            for phi in offsets:
                # 4mm 격자 슬라이스 k의 원본 좌표(mm): (k+z0)*4 + phi → 원본 인덱스 x=(mm)/sz
                zmm = (np.arange(80) + z0) * ZMM + phi
                xs = zmm / sz
                lo = np.clip(np.floor(xs).astype(int), 0, Dn - 1); hi = np.clip(lo + 1, 0, Dn - 1)
                w = np.clip(xs - lo, 0, 1).astype(np.float32)
                vol4 = (u8n[lo] * (1 - w)[:, None, None] + u8n[hi] * w[:, None, None])
                vol4 = np.clip(vol4, 0, 255).astype(np.uint8)
                _, probs = predict_volume(m, vol4, dev, amp=bool(a.amp))   # (10,80,512,512) half(cpu)
                probs = probs.float().numpy()
                # 원본 j → 이 위상 격자 좌표 x4 = (j*sz - phi)/4 - z0
                x4 = (np.arange(Dn) * sz - phi) / ZMM - z0
                l4 = np.clip(np.floor(x4).astype(int), 0, 79); h4 = np.clip(l4 + 1, 0, 79)
                w4 = np.clip(x4 - l4, 0, 1).astype(np.float32)
                pn = probs[:, l4] * (1 - w4)[None, :, None, None] + probs[:, h4] * w4[None, :, None, None]
                pn[:, (x4 < -0.5) | (x4 > 79.5)] = 0.0; pn[0, (x4 < -0.5) | (x4 > 79.5)] = 1.0
                acc = pn if acc is None else acc + pn
                del probs
            pred_n = acc.argmax(0).astype(np.uint8)
            gt_liver = np.isin(lab_n, [1, 2, 5]); gt_flr = lab_n == 2
            if gt_flr.sum() == 0 or gt_liver.sum() == 0: log(f"  skip {pid}: GT 없음"); continue
            pr_liver = pred_n > 0; vox = float(np.prod(sp_g)) / 1000.0
            sel = [c for c in range(1, 10) if (pred_n == c).sum() > 0 and ((pred_n == c) & gt_flr).sum() / (pred_n == c).sum() > 0.5]
            pr_flr = np.isin(pred_n, sel)
            r = {"case": pid, "sz_native": round(sp_g[0], 2),
                 "liver_dice": _dice(pr_liver, gt_liver), "flr_dice": _dice(pr_flr, gt_flr) if sel else 0.0,
                 "flr_hd95_mm": hd95(pr_flr, gt_flr, sp_g) if (sel and not a.no_hd) else float("nan"),
                 "flr_vol_gt_ml": float(gt_flr.sum() * vox), "flr_vol_pred_ml": float(pr_flr.sum() * vox),
                 "flr_pct_gt": float(gt_flr.sum() / gt_liver.sum() * 100), "flr_pct_pred": float(pr_flr.sum() / max(pr_liver.sum(), 1) * 100)}
            rows.append(r)
            log(f"  {pid}: FLR {r['flr_dice']:.4f} (liver {r['liver_dice']:.4f})")
        except Exception as e:
            log(f"  FAIL {pid}: {type(e).__name__}: {str(e)[:120]}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "metrics.csv"), index=False)
    dd = df.flr_pct_pred - df.flr_pct_gt
    s = {"n": len(df), "phases": a.phases, "liver_dice": round(float(df.liver_dice.mean()), 4),
         "flr_dice": round(float(df.flr_dice.mean()), 4), "flr_dice_median": round(float(df.flr_dice.median()), 4),
         "flr_hd95_mm": round(float(df.flr_hd95_mm.mean()), 2) if not a.no_hd else None,
         "flr_pct_bias_pp": round(float(dd.mean()), 2), "flr_pct_sd_pp": round(float(dd.std()), 2)}
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()
