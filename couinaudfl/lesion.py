"""병변-분절 위치 분석 — 센터 로컬에서 병변별 겹침표(익명 숫자만)를 만든다.

행 단위 = 병변 연결성분 하나. GT 분절(ch1–9)과 예측 라벨맵 각각에 대해
  frac(s) = (병변 ∩ 분절 s) / 병변 부피
를 기록한다. "병변이 분절 s에 있다" 판정은 이 비율에 임계치를 걸어 하며(서버측 분석에서 스윕),
GT 기준은 gt_frac, 모델 기준은 pred_frac 이다.
"""
from __future__ import annotations
import os
import numpy as np
from scipy import ndimage
from .data import SEG_NAMES, NUM_CLASSES

MIN_LESION_ML = 0.05   # 이보다 작은 성분은 잡음으로 간주


def load_lesions(case_dir: str):
    """mask.npy ch10–17 → (병변 union bool (D,H,W), 채널 인덱스맵) 또는 None(병변 채널 없음)."""
    p = os.path.join(case_dir, "mask.npy")
    if not os.path.exists(p): return None
    m = np.load(p, mmap_mode="r")
    if m.shape[0] < 11: return None
    les = np.asarray(m[10:min(18, m.shape[0])])
    if les.sum() == 0: return None
    ch = np.zeros(les.shape[1:], np.int16)
    for k in range(les.shape[0]):
        ch[les[k] > 0] = 10 + k
    return les.any(0), ch


def lesion_overlap_rows(case: str, case_dir: str, gt_label: np.ndarray, pred_label: np.ndarray, spacing) -> list[dict]:
    lz = load_lesions(case_dir)
    if lz is None: return []
    les, chmap = lz
    D = min(les.shape[0], gt_label.shape[0], pred_label.shape[0])
    les, chmap = les[:D], chmap[:D]; gt, pr = gt_label[:D], pred_label[:D]
    vox_ml = float(np.prod(spacing)) / 1000.0
    comp, nc = ndimage.label(les)
    rows = []
    for li in range(1, nc + 1):
        sel = comp == li; v = int(sel.sum())
        if v * vox_ml < MIN_LESION_ML: continue
        r = {"case": case, "lesion_id": li, "lesion_ml": round(v * vox_ml, 3),
             "lesion_channel": int(np.bincount(chmap[sel]).argmax())}
        g = gt[sel]; p = pr[sel]
        # 중심(질량중심에 가장 가까운 병변 복셀) 기준 단일 분절 지정 — 임상적 '위치' 정의
        idx = np.argwhere(sel); com = idx.mean(0); ci = idx[np.argmin(((idx - com) ** 2).sum(1))]
        r["com_gt_seg"] = int(gt[tuple(ci)]); r["com_pred_seg"] = int(pr[tuple(ci)])
        for c in range(1, NUM_CLASSES):
            n = SEG_NAMES[c - 1]
            r[f"gt_frac_{n}"] = round(float((g == c).sum() / v), 4)
            r[f"pred_frac_{n}"] = round(float((p == c).sum() / v), 4)
        rows.append(r)
    return rows
