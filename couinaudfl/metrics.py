"""환자·분절별 지표 — Dice, HD95(mm), 부피(mL, GT·예측), 전체 간. 결과는 long-format 행 리스트."""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from .data import SEG_NAMES, NUM_CLASSES

_MAP10TO9 = np.array([0, 1, 2, 3, 4, 4, 5, 6, 7, 8]); SEG8_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]


def _surface(mask: np.ndarray) -> np.ndarray:
    er = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~er


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """95% Hausdorff (mm). 한쪽이 비면 nan. 거리변환 기반(대용량 안전)."""
    if not pred.any() or not gt.any(): return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~gt, sampling=spacing); dt_p = ndimage.distance_transform_edt(~pred, sampling=spacing)
    d1 = dt_g[sp]; d2 = dt_p[sg]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return float(2 * (pred & gt).sum() / s) if s > 0 else float("nan")


def case_metrics(pred: np.ndarray, gt: np.ndarray, spacing, case: str, seg8: bool = False, with_hd: bool = True) -> list[dict]:
    """pred/gt (D,H,W) uint8 라벨맵(10클래스). seg8=True면 4a/4b 병합 후 8분절 기준으로 계산."""
    vox_ml = float(np.prod(spacing)) / 1000.0
    if seg8:
        pred, gt = _MAP10TO9[pred], _MAP10TO9[gt]; names = SEG8_NAMES
    else:
        names = SEG_NAMES
    rows = []
    for c, name in enumerate(names, start=1):
        p = pred == c; g = gt == c
        rows.append({"case": case, "segment": name, "dice": dice(p, g), "hd95_mm": hd95(p, g, spacing) if with_hd else float("nan"),
                     "vol_gt_ml": float(g.sum() * vox_ml), "vol_pred_ml": float(p.sum() * vox_ml)})
    p = pred > 0; g = gt > 0
    rows.append({"case": case, "segment": "liver", "dice": dice(p, g), "hd95_mm": hd95(p, g, spacing) if with_hd else float("nan"),
                 "vol_gt_ml": float(g.sum() * vox_ml), "vol_pred_ml": float(p.sum() * vox_ml)})
    return rows


def summarize(rows: list[dict]) -> dict:
    import pandas as pd
    df = pd.DataFrame(rows); seg = df[df.segment != "liver"]
    per = seg.groupby("segment").dice.mean()
    return {"dice_mean_segments": float(seg.dice.mean()), "dice_per_segment": per.round(4).to_dict(),
            "hd95_mean_segments": float(seg.hd95_mm.mean()), "dice_liver": float(df[df.segment == "liver"].dice.mean()),
            "n_cases": int(df.case.nunique())}
