#!/usr/bin/env python
"""환자 카탈로그 — 센터 전 환자의 메타데이터·GT 부피를 한 번에 기록(추후 재방문 불필요).

  python scripts/patient_catalog.py --data D:\data\liver --exp exp4c --site A
→ outputs/<exp>/client_<site>/patient_catalog.csv  (행 = 환자; 실제 ID 포함, 내보내기 시 Case N으로 치환)
열: case, n_slices, spacing_z/y/x(mm), voxel_ml, liver_ml, seg{S1..S8}_ml, seg_present_count, lesion_ch10..17_ml, lesion_total_ml,
    liver_z0, liver_z1(간 포함 슬라이스 범위), liver_median_u8, image_mean_u8, fold_test(어느 fold의 test인지)
"""
import os, sys, json, argparse, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, yaml
from couinaudfl.data import list_cases, load_case, kfold_split, SEG_NAMES


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--exp", required=True); ap.add_argument("--site", default=socket.gethostname())
    ap.add_argument("--config", default=None); ap.add_argument("--out", default="outputs"); ap.add_argument("--run", required=True); a = ap.parse_args()
    C = yaml.safe_load(open(a.config or f"configs/{a.exp}.yaml")); root = os.path.join(a.out, a.exp, a.run, f"client_{a.site}"); os.makedirs(root, exist_ok=True)
    cases = list_cases(a.data); fold_of = {}
    for k in range(C.get("n_folds", 5)):
        _, _, te = kfold_split(cases, k, C.get("n_folds", 5), C.get("seed", 42), C.get("val_frac", 0.1))
        for c in te: fold_of[os.path.basename(c)] = k
    rows = []
    for cd in cases:
        img, lab, meta = load_case(cd); name = os.path.basename(cd)
        sp = meta.get("spacing") or meta.get("orig_spacing") or [4.0, 0.73046875, 0.73046875]; sz, sy, sx = 4.0, float(sp[1]), float(sp[2]); vox = sz * sy * sx / 1000.0
        r = {"case": name, "n_slices": int(img.shape[0]), "spacing_z": sz, "spacing_y": sy, "spacing_x": sx, "spacing_source": meta.get("spacing_source", "?"), "voxel_ml": vox, "liver_ml": float((lab > 0).sum() * vox)}
        for c, n in enumerate(SEG_NAMES, start=1): r[f"seg{n}_ml"] = float((lab == c).sum() * vox)
        r["seg_present_count"] = int(sum(r[f"seg{n}_ml"] > 0 for n in SEG_NAMES))
        mp = os.path.join(cd, "mask.npy")
        try:
            m = np.load(mp, mmap_mode="r"); tot = 0.0
            for ch in range(10, min(18, m.shape[0])):
                v = float(np.asarray(m[ch]).sum() * vox); r[f"lesion_ch{ch}_ml"] = v; tot += v
            r["lesion_total_ml"] = tot
        except Exception: r["lesion_total_ml"] = float("nan")
        zs = np.nonzero((lab > 0).any((1, 2)))[0]; r["liver_z0"] = int(zs[0]) if len(zs) else -1; r["liver_z1"] = int(zs[-1]) if len(zs) else -1
        r["liver_median_u8"] = float(np.median(img[lab > 0])) if (lab > 0).any() else float("nan"); r["image_mean_u8"] = float(img.mean())
        r["fold_test"] = fold_of.get(name, -1); rows.append(r)
    # 병변 채널 매핑 우선순위: ① 데이터 폴더(센터 수정본) ② 레포 내장 configs/lesion_labels/<site>.json ③ 템플릿
    import shutil
    lj = os.path.join(a.data, "lesion_labels.json"); bj = os.path.join("configs", "lesion_labels", f"{a.site}.json")
    if os.path.exists(lj):
        shutil.copy(lj, os.path.join(root, "lesion_labels.json")); print(f"병변 매핑: 데이터 폴더 제공본 사용 ({lj})")
    elif os.path.exists(bj):
        shutil.copy(bj, os.path.join(root, "lesion_labels.json")); print(f"병변 매핑: 레포 내장 {a.site} 매핑 사용 ({bj})")
    else:
        print("병변 매핑: 없음 — 채널 번호로 기록(TEMPLATE 생성)")
        json.dump({"_설명": "병변 채널(10–17) → 유형 매핑. 센터에서 채워 데이터 폴더에 lesion_labels.json 으로 저장", "10": "", "11": "", "12": "", "13": "", "14": "", "15": "", "16": "", "17": ""},
                  open(os.path.join(root, "lesion_labels_TEMPLATE.json"), "w"), ensure_ascii=False, indent=1)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(root, "patient_catalog.csv"), index=False)
    print(f"patient_catalog.csv: {len(df)}명 → {root}")


if __name__ == "__main__":
    main()
