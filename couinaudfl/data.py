"""데이터 계층 — 기관 포맷(image.npy uint8 (D,512,512) + mask.npy (C,D,512,512)) 전용.

라벨 규약 (10클래스 소프트맥스): 0 배경, 1 S1, 2 S2, 3 S3, 4 S4a, 5 S4b, 6 S5, 7 S6, 8 S7, 9 S8
  - 기관 mask.npy: 18채널(ch0 배경, ch1–9 분절, ch10–17 병변) → ch0..9 argmax, 병변 채널은 무시
  - 공용 mask.npy: 10채널(ch0 배경, ch1–9) — 8분절 데이터는 ch5(S4b)가 비어 있고 ch4가 S4 전체.
    이런 케이스는 meta.json의 segments==8 → 손실에서 4a/4b 확률을 합쳐 S4 하나로 감독한다(partial-label).
학습 패치: z 전체(80) × 256 × 256, 간 포함 위치를 우선 표본화. 좌우 반전 증강은 금지(퀴노는 키랄).
"""
from __future__ import annotations
import os, json, random
import numpy as np
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 10
SEG_NAMES = ["S1", "S2", "S3", "S4a", "S4b", "S5", "S6", "S7", "S8"]
PATCH = (96, 256, 256)   # TS 구조는 5회 다운샘플(÷32) → z 80은 96으로 zero-pad (볼륨은 80슬라이스 그대로)
MEAN, STD = 0.5, 0.25          # uint8/255 기준 고정 정규화 (기관·공용 동일 윈도우 WL80/WW225)


def normalize(img_u8: np.ndarray) -> np.ndarray:
    return ((img_u8.astype(np.float32) / 255.0) - MEAN) / STD


DEFAULT_SPACING = [4.0, 0.73046875, 0.73046875]   # z 4mm 고정, 면내는 분당 기준값(센터별 spacing.csv로 대체 권장)
_SPACING_CACHE: dict = {}


SPACING_CSV = os.environ.get("COUINAUD_SPACING_CSV", "")   # 데이터 루트 밖의 파일을 쓸 때(예비)


def set_spacing_file(path: str):
    """--spacing-file 인자용: 이후 모든 load_case가 이 파일(csv/xlsx)을 최우선으로 사용."""
    global SPACING_CSV
    SPACING_CSV = path or SPACING_CSV
    _SPACING_CACHE.clear()


def _norm_col(c):  # 열 이름 정규화 (Resolution (512, 1024) → resolution 등)
    c = str(c).strip().lower()
    for key, names in {"id": ["id", "case", "patient_id"], "resolution": ["resolution"], "pixel": ["pixel spacing", "pixel_spacing", "pixelspacing"],
                       "thick": ["slice thickeness", "slice thickness", "thickness"], "incr": ["incremental", "increment"]}.items():
        if any(c.startswith(n) for n in names): return key
    return c


def _rows_from_xlsx(path):
    import pandas as pd
    df = pd.read_excel(path, dtype=str)   # ID 선행 0·문자 보존
    df.columns = [_norm_col(c) for c in df.columns]
    return df.to_dict("records")


def _spacing_table(root: str) -> dict:
    """환자별 spacing 표 → {case: [z, s, s]} (512 기준 면내 s, 환자별 z).
    파일 탐색: $COUINAUD_SPACING_CSV → <data>/spacing.(xlsx|csv) → <data>/volumes_per_patient.csv(v1) → <data>/*.xlsx 1개.
    xlsx 열(센터 시트 원형): Matching, ID, Resolution(512|1024), Pixel spacing, Slice thickeness, Incremental[, Date]
      · 면내 s = Pixel spacing × Resolution/512 (1024 매트릭스 ×2 보정)
      · z = Incremental → 없으면 Slice thickness → 없으면 4.0 (강릉 빈칸 규칙)
      · ID는 문자열 그대로 매칭(선행 0·문자 보존), 폴더명과 정확 일치 우선 + 숫자화 보조 매칭"""
    if root in _SPACING_CACHE: return _SPACING_CACHE[root]
    cands = [SPACING_CSV, os.path.join(root, "spacing.xlsx"), os.path.join(root, "spacing.csv"), os.path.join(root, "volumes_per_patient.csv")]
    import glob as _g
    xs = [p for p in _g.glob(os.path.join(root, "*.xlsx")) if not os.path.basename(p).startswith("~")]
    if len(xs) == 1: cands.append(xs[0])
    tbl = {}
    for p in cands:
        if not p or not os.path.exists(p): continue
        rows = []
        if p.lower().endswith(".xlsx"):
            try: rows = _rows_from_xlsx(p)
            except Exception: continue
        else:
            import csv
            rows = [{ _norm_col(k): v for k, v in r.items()} for r in csv.DictReader(open(p, encoding="utf-8-sig"))]
        for r in rows:
            k = str(r.get("id") or "").strip()
            if not k or k.lower() == "nan" or k in tbl: continue
            try:
                def num(key):
                    v = r.get(key)
                    v = "" if v is None else str(v).strip()
                    if not v or v.lower() == "nan": return None
                    try: return float(v)
                    except ValueError: return None   # '?', '100W116 (...)' 등 비수치 → 결측
                if r.get("spacing_y"):   # 단순 csv 형식
                    tbl[k] = [num("spacing_z") or 4.0, float(r["spacing_y"]), float(r["spacing_x"])]; continue
                px = num("pixel") if "pixel" in r else num("pixel_spacing")
                res = num("resolution") or num("orig_matrix") or 512.0
                sp_in = (px * res / 512.0) if px is not None else None
                if sp_in is not None and not (0.3 <= sp_in <= 1.5): sp_in = None   # '2', 문자열 등 비정상 → 결측 처리
                z = num("incr") or num("thick") or 4.0
                if not (1.0 <= z <= 10.0): z = 4.0
                if sp_in is None and z == 4.0 and (px is not None):   # 면내 무효 + z 기본 → 완전 무효로 보되 z 정보는 이미 반영됨
                    pass
                tbl[k] = [z, sp_in if sp_in is not None else -1.0, sp_in if sp_in is not None else -1.0]
            except (KeyError, ValueError, TypeError): continue
        if tbl: break
    # 주의: ID는 문자열 '정확 일치'만 사용 — 숫자부 동일·접미문자(예 …a1)로 구별되는 환자가 존재하므로
    # 선행 0 제거 등 근사 매칭은 오배정 위험이 있어 하지 않는다. 미매칭은 spacing_source=default로 드러남.
    _SPACING_CACHE[root] = tbl; return tbl


LABEL_CACHE = os.environ.get("COUINAUD_LABEL_CACHE", "")   # 기관 데이터 폴더에 쓰기 싫을 때 캐시 위치 지정


def _label_path(case_dir: str) -> str:
    if LABEL_CACHE:
        p = os.path.join(LABEL_CACHE, os.path.basename(os.path.dirname(case_dir.rstrip("/"))), os.path.basename(case_dir.rstrip("/")))
        os.makedirs(p, exist_ok=True); return os.path.join(p, "label.npy")
    return os.path.join(case_dir, "label.npy")


def load_case(case_dir: str):
    """→ (image uint8 (D,H,W), label uint8 (D,H,W), meta dict).
    mask.npy(10~18채널, 케이스당 210–380MB)는 처음 한 번만 argmax해서 label.npy(21MB)로 캐시한다 — I/O 병목 제거."""
    img = np.load(os.path.join(case_dir, "image.npy"), mmap_mode="r")
    mp = os.path.join(case_dir, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    if "spacing" not in meta:   # 부피(mL)용 spacing: ① 환자폴더 meta.json ② 데이터 루트 spacing.csv ③ 기본값(분당 0.73mm)
        sp = _spacing_table(os.path.dirname(case_dir.rstrip("/"))).get(os.path.basename(case_dir.rstrip("/")))
        if sp is not None and sp[1] > 0:
            meta["spacing"] = [float(sp[0]), float(sp[1]), float(sp[2])]; meta["spacing_source"] = "spacing_table"
        elif sp is not None:   # 시트에 있으나 면내 값이 무효(예: '2', 문자열) → z만 시트값, 면내는 기본값
            meta["spacing"] = [float(sp[0]), DEFAULT_SPACING[1], DEFAULT_SPACING[2]]; meta["spacing_source"] = "table_invalid_inplane"
        elif "orig_spacing" in meta: meta["spacing"] = [4.0, float(meta["orig_spacing"][1]), float(meta["orig_spacing"][2])]; meta["spacing_source"] = "meta.orig_spacing"   # 공용 전처리본은 z=4mm 리샘플 확정
        else: meta["spacing"] = DEFAULT_SPACING; meta["spacing_source"] = "default"
    else: meta.setdefault("spacing_source", "meta.json")
    lp = _label_path(case_dir)
    if os.path.exists(lp):
        lab = np.load(lp)
    else:
        msk = np.load(os.path.join(case_dir, "mask.npy"), mmap_mode="r")
        lab = np.argmax(np.asarray(msk[:NUM_CLASSES]), axis=0).astype(np.uint8)
        try: np.save(lp, lab)
        except OSError: pass
    D = min(img.shape[0], lab.shape[0])
    return np.asarray(img[:D]), lab[:D], meta


def is_8seg(meta: dict) -> bool:
    return int(meta.get("segments", 9)) == 8


class CouinaudDataset(Dataset):
    """mode='train': 무작위 패치 + 증강 / 'eval': 전체 볼륨(슬라이딩 윈도우용)."""

    def __init__(self, case_dirs: list[str], mode: str = "train", patch=PATCH, fg_prob: float = 0.8,
                 augment: bool = True, cache: bool = False):
        self.cases = list(case_dirs); self.mode = mode; self.patch = tuple(patch)
        self.fg_prob = fg_prob; self.augment = augment and mode == "train"
        self._cache = {} if cache else None
        self._aug = self._build_aug() if self.augment else None

    def __len__(self): return len(self.cases)

    def _get(self, i):
        if self._cache is not None and i in self._cache: return self._cache[i]
        item = load_case(self.cases[i])
        if self._cache is not None: self._cache[i] = item
        return item

    @staticmethod
    def _build_aug():
        from monai import transforms as T
        return T.Compose([
            T.RandAffined(keys=["image", "label"], prob=0.5, rotate_range=(0, 0, np.deg2rad(10)),
                          scale_range=(0, 0.1, 0.1), mode=("bilinear", "nearest"), padding_mode="zeros"),
            T.RandScaleIntensityd(keys="image", factors=0.15, prob=0.5),
            T.RandShiftIntensityd(keys="image", offsets=0.15, prob=0.5),
            T.RandAdjustContrastd(keys="image", gamma=(0.8, 1.25), prob=0.3),
            T.RandGaussianNoised(keys="image", std=0.05, prob=0.2),
            T.RandGaussianSmoothd(keys="image", sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.0, 0.5), prob=0.15),
        ])

    def _crop(self, img, lab):
        D, H, W = lab.shape; pd, ph, pw = self.patch
        # z: 패치가 볼륨보다 크면 zero-pad, 작으면 무작위 시작
        if random.random() < self.fg_prob and lab.any():
            zs, ys, xs = np.nonzero(lab[::4, ::8, ::8]); k = random.randrange(len(zs))
            cz, cy, cx = zs[k] * 4, ys[k] * 8, xs[k] * 8
            z0 = int(np.clip(cz - pd // 2 + random.randint(-pd // 4, pd // 4), 0, max(D - pd, 0)))
            y0 = int(np.clip(cy - ph // 2 + random.randint(-ph // 4, ph // 4), 0, max(H - ph, 0)))
            x0 = int(np.clip(cx - pw // 2 + random.randint(-pw // 4, pw // 4), 0, max(W - pw, 0)))
        else:
            z0 = random.randint(0, max(D - pd, 0)); y0 = random.randint(0, max(H - ph, 0)); x0 = random.randint(0, max(W - pw, 0))
        ci = img[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]; cl = lab[z0:z0 + pd, y0:y0 + ph, x0:x0 + pw]
        if ci.shape != self.patch:
            pi = np.zeros(self.patch, ci.dtype); pl = np.zeros(self.patch, cl.dtype)
            pi[:ci.shape[0], :ci.shape[1], :ci.shape[2]] = ci; pl[:cl.shape[0], :cl.shape[1], :cl.shape[2]] = cl
            ci, cl = pi, pl
        return ci, cl

    def __getitem__(self, i):
        img, lab, meta = self._get(i)
        if self.mode == "train":
            img, lab = self._crop(img, lab)
        x = normalize(img)[None]; y = lab.astype(np.int64)
        if self._aug is not None:
            d = self._aug({"image": torch.from_numpy(x), "label": torch.from_numpy(y)[None]})
            x = d["image"].float(); y = d["label"][0].long()
        else:
            x = torch.from_numpy(np.ascontiguousarray(x)); y = torch.from_numpy(y)
        return {"image": x, "label": y, "seg8": torch.tensor(is_8seg(meta)), "case": os.path.basename(self.cases[i]),
                "spacing": torch.tensor(meta.get("spacing", meta.get("orig_spacing", [4.0, 0.7, 0.7]))[:3] if meta else [4.0, 0.7, 0.7], dtype=torch.float32)}


def list_cases(root: str) -> list[str]:
    return sorted(os.path.join(root, d) for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and os.path.exists(os.path.join(root, d, "image.npy")))


def kfold_split(ids: list[str], fold: int, n_folds: int = 5, seed: int = 42, val_frac: float = 0.1):
    """정렬 → seed 셔플 → 교차 배정. test=fold k, val=나머지의 val_frac (v2 확정 규약)."""
    ids = sorted(ids); rng = random.Random(seed); rng.shuffle(ids)
    folds = [ids[k::n_folds] for k in range(n_folds)]
    test = folds[fold]; rest = [p for k in range(n_folds) if k != fold for p in folds[k]]
    rng2 = random.Random(seed + 1000 + fold); rng2.shuffle(rest)
    nv = max(1, int(round(len(rest) * val_frac)))
    return sorted(rest[nv:]), sorted(rest[:nv]), sorted(test)
