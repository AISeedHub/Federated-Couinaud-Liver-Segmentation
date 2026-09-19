# -*- coding: utf-8 -*-
"""공개 데이터 1.5mm 얇은 판(public_thin15) 전처리 — E9(얇은 해상도 사전학습)용.
4mm판(public_v2)과 동일 케이스명·동일 분할을 유지한 채, 원본 NIfTI에서 z=1.5mm로 재구성.
산출: <root>/<dataset>/<case>/{image.npy(uint8 D,512,512), label.npy(uint8), meta.json}
라벨 값 매핑은 하드코딩하지 않고, 케이스별로 기존 4mm label.npy와의 z-대응 다수결로 유도·검증한다
(데이터셋당 첫 3례 매핑이 일치해야 진행 — 불일치 시 중단).
사용: .venv/bin/python scripts/prep_thin15.py --out /data/campaign/public_thin15 --workers 8
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = "/data/datasets/couinaud_public"
V2 = "/data/campaign/public_v2"
ZT = 1.5
WL, WW = 80.0, 225.0


def dicom_zyx(d):
    import pydicom, glob as _g
    fs=[pydicom.dcmread(f) for f in _g.glob(f"{d}/image_*")+_g.glob(f"{d}/*.dcm")]
    fs=[x for x in fs if hasattr(x,"ImagePositionPatient")]
    fs.sort(key=lambda x: -float(x.ImagePositionPatient[2]))
    vol=np.stack([x.pixel_array.astype(np.float32)*float(getattr(x,"RescaleSlope",1))+float(getattr(x,"RescaleIntercept",0)) for x in fs])
    sz=abs(float(fs[0].ImagePositionPatient[2])-float(fs[1].ImagePositionPatient[2]))
    sy,sx=map(float,fs[0].PixelSpacing)
    return vol,(sz,sy,sx)


def nii_zyx(path):
    """NIfTI → (arr (D,H,W) 두부→미부, spacing (sz,sy,sx)). RAS 캐노니컬 후 z 내림차순."""
    im = nib.as_closest_canonical(nib.load(path))
    a = np.asanyarray(im.dataobj)
    a = np.transpose(a, (2, 1, 0))          # (z,y,x); RAS z는 상향 → 두부→미부로 뒤집기
    a = a[::-1]
    a = a[:, ::-1, ::-1]                     # y·x 반전 — 4mm판과의 슬라이스 대조로 검증
    z = im.header.get_zooms()
    return np.ascontiguousarray(a), (float(z[2]), float(z[1]), float(z[0]))


def sources(ds, case):
    if ds == "01_msd08_tian":
        ip = f"{SRC}/Task08_HepaticVessel/imagesTr/{case}.nii.gz"
        if not os.path.exists(ip): ip = f"{SRC}/Task08_HepaticVessel/imagesTs/{case}.nii.gz"
        return (ip, f"{SRC}/couinaud_annotation/{case}.nii.gz")
    if ds == "02_msd08_nih":
        ip = f"{SRC}/Task08_HepaticVessel/imagesTr/{case}.nii.gz"
        if not os.path.exists(ip): ip = f"{SRC}/Task08_HepaticVessel/imagesTs/{case}.nii.gz"
        return (ip, f"{SRC}/nih_labels/labels/{case}.nii.gz")
    if ds == "03_lits_zhang":
        n = int(case.split("_")[1])
        return (f"{SRC}/Task03_Liver/imagesTr/liver_{n}.nii.gz", "/data/datasets/zhang_couinaud_labels/Datasets/LiTS/label/LiTS-%d.nii.gz" % n)
    if ds == "04_ircadb_zhang":
        n = int(case.split("_")[1])
        return (None, "/data/datasets/zhang_couinaud_labels/Datasets/3Dircadb/label/3Dircadb-%d.nii.gz" % n)  # 이미지 소스 별도
    raise KeyError(ds)


def derive_map(lab_thin, sz_thin, case_dir_v2, min_frac=0.8):
    """thin 라벨 값 → 4mm판 클래스 매핑을 z-대응 다수결로 유도."""
    meta = json.load(open(f"{case_dir_v2}/meta.json")); z0 = int(meta.get("z0", 0))
    lab4 = np.load(f"{case_dir_v2}/label.npy")
    m = {}
    ratio = 4.0 / sz_thin
    for k in range(0, lab4.shape[0], 2):
        j = int(round((k + z0) * ratio))
        if j >= lab_thin.shape[0]: break
        a4 = lab4[k]; at = lab_thin[j]
        if at.shape != a4.shape: return None
        for v in np.unique(at):
            if v == 0: continue
            sel = at == v
            if sel.sum() < 500: continue
            vals, cnt = np.unique(a4[sel], return_counts=True)
            tgt = int(vals[cnt.argmax()])
            m.setdefault(int(v), []).append(tgt)
    out = {}
    for v, lst in m.items():
        vals, cnt = np.unique(lst, return_counts=True)
        if cnt.max() / cnt.sum() < min_frac: return None   # 애매하면 실패 처리
        out[v] = int(vals[cnt.argmax()])
    return out


def process(ds, case, out_root, vmap):
    od = f"{out_root}/{ds}/{case}"
    if os.path.exists(f"{od}/label.npy"): return "skip"
    ip, lp = sources(ds, case)
    if ds == "04_ircadb_zhang":
        n = int(case.split("_")[1])
        ip = f"{SRC}/_ircad/3Dircadb1/3Dircadb1.{n}/PATIENT_DICOM"
    if ip is None or not os.path.exists(ip) or not os.path.exists(lp): return "nosrc"
    img, sp = (dicom_zyx(ip) if os.path.isdir(ip) else nii_zyx(ip)); lab, spl = nii_zyx(lp)
    if img.shape[0] != lab.shape[0] and abs(img.shape[0]-lab.shape[0])<=2:
        D=min(img.shape[0],lab.shape[0]); img,lab=img[:D],lab[:D]
    if img.shape != lab.shape: return "shape"
    if img.shape[1:] != (512, 512): return "inplane"
    sz = sp[0]
    # z-crop: 라벨 z범위 ±20mm
    zz = np.where((lab > 0).any(axis=(1, 2)))[0]
    if len(zz) == 0: return "empty"
    pad = int(round(20.0 / sz))
    z1, z2 = max(0, zz[0] - pad), min(lab.shape[0], zz[-1] + pad + 1)
    img, lab = img[z1:z2], lab[z1:z2]
    f = sz / ZT
    img_t = zoom(img.astype(np.float32), (f, 1, 1), order=1)
    lab_t = zoom(lab.astype(np.int16), (f, 1, 1), order=0)
    lo, hi = WL - WW / 2, WL + WW / 2
    u8 = np.clip((img_t - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    L = np.zeros(lab_t.shape, np.uint8)
    for v, c in vmap.items(): L[lab_t == v] = c
    os.makedirs(od, exist_ok=True)
    np.save(f"{od}/image.npy", u8); np.save(f"{od}/label.npy", L)
    m4 = json.load(open(f"{V2}/{ds}/{case}/meta.json"))
    meta = {k: m4.get(k) for k in ("source_image", "label_source", "segments", "hash", "lee_test50", "same_image_in", "modality") if k in m4}
    meta.update({"spacing": [ZT, sp[1], sp[2]], "z_mm": ZT, "thin": True, "window": {"WL": WL, "WW": WW},
                 "orig_spacing": [sz, sp[1], sp[2]], "shape": list(u8.shape), "vmap": vmap})
    json.dump(meta, open(f"{od}/meta.json", "w"), indent=1)
    return "ok"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/data/campaign/public_thin15")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--datasets", nargs="*", default=["01_msd08_tian", "02_msd08_nih", "03_lits_zhang", "04_ircadb_zhang"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # 평가 전용 세트(05/06/07)는 4mm판으로 심링크(분할 열거용)
    for ds in ("05_msd08_medseg", "06_ts_mr", "07_crlm"):
        dst = f"{a.out}/{ds}"
        if not os.path.exists(dst): os.symlink(f"{V2}/{ds}", dst)
    from concurrent.futures import ProcessPoolExecutor
    for ds in a.datasets:
        cases = sorted(os.listdir(f"{V2}/{ds}"))
        cases = [c for c in cases if os.path.isdir(f"{V2}/{ds}/{c}")]
        # 매핑 유도(첫 3례 일치 필수)
        maps = []
        for c in cases[:6]:
            ip, lp = sources(ds, c)
            if ds == "04_ircadb_zhang":
                n = int(c.split("_")[1])
                ip = f"{SRC}/_ircad/3Dircadb1/3Dircadb1.{n}/PATIENT_DICOM"
            if not (ip and os.path.exists(ip) and os.path.exists(lp)): continue
            lab, sp = nii_zyx(lp)
            m = derive_map(lab, sp[0], f"{V2}/{ds}/{c}")
            if m: maps.append((c, m))
            if len(maps) >= 3: break
        assert len(maps) >= 2, f"{ds}: 매핑 유도 실패 {maps}"
        base = maps[0][1]
        for c, m in maps[1:]:
            assert m == base, f"{ds}: 매핑 불일치 {maps[0][0]}{base} vs {c}{m}"
        print(f"[{ds}] vmap={base} ({len(cases)} cases)", flush=True)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(process, ds, c, a.out, base): c for c in cases}
            stat = {}
            for f_ in futs:
                r = f_.result(); stat[r] = stat.get(r, 0) + 1
        print(f"[{ds}] {stat}", flush=True)


if __name__ == "__main__":
    main()
