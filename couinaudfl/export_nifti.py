"""기관 포맷(uint8, WL80/WW225) → 근사 HU NIfTI 복원. 공개 모델(TS 570 등) zero-shot 평가 입력용.

HU ≈ v/255·WW + (WL − WW/2). 윈도우 밖(−32.5 미만, 192.5 초과)은 포화되어 복원 불가 → 평가 한계로 명시.
축: 기관 (Z cranial→caudal, Y anterior→posterior, X right→left) → RAS canonical: inst_from_canonical의 역변환.
"""
from __future__ import annotations
import numpy as np, nibabel as nib

WL, WW = 80.0, 225.0


def u8_to_hu(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32) / 255.0 * WW + (WL - WW / 2)


def inst_to_canonical(a: np.ndarray) -> np.ndarray:
    """inst (Z,Y,X) → canonical RAS (X,Y,Z) 배열 (prep의 inst_from_canonical 역변환)."""
    return np.transpose(a, (2, 1, 0))[::-1, ::-1, ::-1]


def save_nifti(arr_inst: np.ndarray, spacing_zyx, path: str, dtype=np.float32):
    sz, sy, sx = spacing_zyx; can = np.ascontiguousarray(inst_to_canonical(arr_inst)).astype(dtype)
    aff = np.diag([sx, sy, sz, 1.0]); nib.save(nib.Nifti1Image(can, aff), path)


def load_nifti_as_inst(path: str) -> np.ndarray:
    c = nib.as_closest_canonical(nib.load(path)); a = np.asarray(c.dataobj)
    return np.transpose(a[::-1, ::-1, ::-1], (2, 1, 0))
