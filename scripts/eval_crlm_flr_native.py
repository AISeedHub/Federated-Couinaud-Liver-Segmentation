#!/usr/bin/env python
"""CRLM FLR — 원본 격자 재평가.
4mm 학습 격자에서 얻은 10클래스 소프트맥스 확률맵을 z 선형보간으로 원본 슬라이스 위치에 샘플링해 argmax,
원본 DICOM-SEG에서 재구성한 GT와 원본 spacing으로 Dice·HD95·부피를 계산한다(FLR 선택 규칙은 4mm판과 동일).

격자 대응(전처리 역산): 원본 z 인덱스 j → 4mm 전체 인덱스 x = j·f (f=sz/4, zoom과 동일 규약) → 80창 좌표 x−z0.
  python scripts/eval_crlm_flr_native.py --weights outputs/pretrain/best.pth --out outputs/zeroshot/crlm_flr_native
"""
import os, sys, json, glob, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, pandas as pd, pydicom
from couinaudfl.model import build_model
from couinaudfl.data import load_case, list_cases, SEG_NAMES
from couinaudfl.infer import predict_volume
from couinaudfl.metrics import dice as _dice, hd95

TMP = "/data/campaign/_crlm_tmp"; ZMM = 4.0


def native_gt(pid):
    d = os.path.join(TMP, pid)
    dss = [pydicom.dcmread(f, stop_before_pixels=True) for f in glob.glob(f"{d}/ct/**/*.dcm", recursive=True)]
    dss = sorted([x for x in dss if hasattr(x, "ImagePositionPatient")], key=lambda x: -float(x.ImagePositionPatient[2]))
    sz = abs(float(dss[0].ImagePositionPatient[2]) - float(dss[1].ImagePositionPatient[2])); sy, sx = map(float, dss[0].PixelSpacing)
    D = len(dss); zpos = {round(float(x.ImagePositionPatient[2]), 1): i for i, x in enumerate(dss)}
    sg = pydicom.dcmread(glob.glob(f"{d}/seg/**/*.dcm", recursive=True)[0]); arr = sg.pixel_array
    if arr.ndim == 2: arr = arr[None]
    names = {int(s.SegmentNumber): str(s.SegmentLabel) for s in sg.SegmentSequence}
    def cls(lbl):
        l = lbl.lower()
        if "remnant" in l: return 2
        if "liver" in l: return 1
        if "hepatic" in l: return 3
        if "portal" in l: return 4
        if "tumor" in l or "mass" in l: return 5
        return 0
    H, W = arr.shape[1:]
    lab = np.zeros((D, H, W), np.uint8)
    frames = [(fi, int(fg.SegmentIdentificationSequence[0].ReferencedSegmentNumber), round(float(fg.PlanePositionSequence[0].ImagePositionPatient[2]), 1))
              for fi, fg in enumerate(sg.PerFrameFunctionalGroupsSequence)]
    for target in (1, 2, 3, 4, 5):
        for fi, segnum, zp in frames:
            if cls(names.get(segnum, "")) == target and zp in zpos: lab[zpos[zp]][arr[fi] > 0] = target
    return lab, (sz, sy, sx)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--cases", default="/data/campaign/public_v2/07_crlm")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-hd", action="store_true"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda"); m = build_model().to(dev); sd = torch.load(a.weights, map_location=dev, weights_only=False); m.load_state_dict(sd.get("model", sd)); m.eval()
    cases = [c for c in list_cases(a.cases) if os.path.basename(c) != "CRLM-CT-1075"]
    if a.limit: cases = cases[:a.limit]
    log(f"native-grid CRLM FLR: {len(cases)} cases")
    rows = []
    for i, cd in enumerate(cases):
        pid = os.path.basename(cd)
        try:
            img4, _, meta = load_case(cd); z0 = int(meta.get("z0", 0)); szo = meta["orig_spacing"]; f = float(szo[0]) / ZMM
            lab_n, sp_n = native_gt(pid); Dn = lab_n.shape[0]
            _, probs = predict_volume(m, img4, dev, amp=False)          # (10, 80, 512, 512) half
            probs = probs.float().numpy()
            # 원본 z 인덱스 j → 80창 좌표 x = j*f - z0 ; 창 밖(<0 or >79)은 배경으로
            xs = np.arange(Dn) * f - z0
            lo = np.clip(np.floor(xs).astype(int), 0, 79); hi = np.clip(lo + 1, 0, 79); w = np.clip(xs - lo, 0, 1).astype(np.float32)
            pn = probs[:, lo] * (1 - w)[None, :, None, None] + probs[:, hi] * w[None, :, None, None]
            pred_n = pn.argmax(0).astype(np.uint8); pred_n[(xs < -0.5) | (xs > 79.5)] = 0
            if lab_n.shape[1:] != pred_n.shape[1:]:   # in-plane 크기 상이 케이스는 건너뜀(전처리에서 리사이즈된 경우)
                log(f"  skip {pid}: in-plane {lab_n.shape[1:]} != 512"); continue
            gt_liver = np.isin(lab_n, [1, 2, 5]); gt_flr = lab_n == 2
            if gt_flr.sum() == 0 or gt_liver.sum() == 0: log(f"  skip {pid}: GT 없음"); continue
            pr_liver = pred_n > 0; vox = float(np.prod(sp_n)) / 1000.0
            sel = [c for c in range(1, 10) if (pred_n == c).sum() > 0 and ((pred_n == c) & gt_flr).sum() / (pred_n == c).sum() > 0.5]
            pr_flr = np.isin(pred_n, sel)
            r = {"case": pid, "sz_native": round(sp_n[0], 2),
                 "liver_dice": _dice(pr_liver, gt_liver), "flr_dice": _dice(pr_flr, gt_flr) if sel else 0.0,
                 "flr_hd95_mm": hd95(pr_flr, gt_flr, sp_n) if (sel and not a.no_hd) else float("nan"),
                 "flr_vol_gt_ml": float(gt_flr.sum() * vox), "flr_vol_pred_ml": float(pr_flr.sum() * vox),
                 "flr_pct_gt": float(gt_flr.sum() / gt_liver.sum() * 100), "flr_pct_pred": float(pr_flr.sum() / max(pr_liver.sum(), 1) * 100)}
            r["flr_vol_err_pct"] = (r["flr_vol_pred_ml"] - r["flr_vol_gt_ml"]) / r["flr_vol_gt_ml"] * 100
            rows.append(r)
        except Exception as e:
            log(f"  FAIL {pid}: {type(e).__name__}: {str(e)[:100]}"); continue
        if (i + 1) % 20 == 0:
            df = pd.DataFrame(rows); log(f"  {i+1}/{len(cases)} | liver {df.liver_dice.mean():.4f} FLR {df.flr_dice.mean():.4f} HD {df.flr_hd95_mm.mean():.1f}")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "metrics_flr_native.csv"), index=False)
    dd = df.flr_pct_pred - df.flr_pct_gt
    s = {"n": len(df), "liver_dice": round(float(df.liver_dice.mean()), 4), "flr_dice": round(float(df.flr_dice.mean()), 4),
         "flr_dice_median": round(float(df.flr_dice.median()), 4), "flr_hd95_mm": round(float(df.flr_hd95_mm.mean()), 1),
         "flr_vol_abs_err_pct_mean": round(float(df.flr_vol_err_pct.abs().mean()), 2),
         "flr_pct_bias_pp": round(float(dd.mean()), 2), "flr_pct_sd_pp": round(float(dd.std()), 2),
         "flr_vol_corr_pearson": round(float(df.flr_vol_gt_ml.corr(df.flr_vol_pred_ml)), 4)}
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()
