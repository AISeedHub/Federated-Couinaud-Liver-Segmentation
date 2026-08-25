# -*- coding: utf-8 -*-
"""

출력 (outputs/analysis/):
  volume_comparison.csv / spacing_calibration.csv / segment_stats.csv
  disease_volume.csv / lobe_ratio.csv / lesion_atrophy.csv / flr.csv
"""
import os, csv, math
import numpy as np

# ===== 경로 =====
REPO     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = "data/combined_80"
ISP_CSV  = os.path.join(REPO, "segment_volumes.csv")
OUT_DIR  = os.path.join(REPO, "outputs/analysis")

# ===== ISP CSV 컬럼 → 리스트 index(0..8 = Seg1..Seg8) =====
ISP_TO_CH = {
    "segment_1_cc": 0, "segment_2_cc": 1, "segment_3_cc": 2,
    "segment_4A_cc": 3, "segment_4B_cc": 4, "segment_5_cc": 5,
    "segment_6_cc": 6, "segment_7_cc": 7, "segment_8_cc": 8,
}
SEG_NAMES = ["Seg1","Seg2","Seg3","Seg4a","Seg4b","Seg5","Seg6","Seg7","Seg8"]

# 병변 채널 (원본 18채널, ch10~17)
LESION_CH = {
    10: "cyst", 11: "tumor", 12: "tace_site", 13: "metallic",
    14: "ablation", 15: "op_site", 16: "lipiodol", 17: "surgical_clip",
}

RIGHT = ["Seg5","Seg6","Seg7","Seg8"]
LEFT  = ["Seg2","Seg3","Seg4a","Seg4b"]
CAUD  = ["Seg1"]

N_SEG   = 9
SLICE_Z = 4.0   # thick=incr=4mm

# 실측 in-plane 픽셀 스페이싱은 PHI 보호를 위해 배포하지 않습니다.
# configs/spacing.csv (columns: patient_id,xy_spacing_mm) 를 준비하세요.
import csv as _csv
REAL_SPACING = {}
_sp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "spacing.csv")
if os.path.exists(_sp_path):
    with open(_sp_path) as _f:
        for _r in _csv.DictReader(_f):
            REAL_SPACING[_r["patient_id"]] = float(_r["xy_spacing_mm"])
else:
    print("[WARN] configs/spacing.csv 없음 - ISP 역산 fallback만 사용됩니다.")

# 원본 1024 매트릭스 (전처리로 512 다운샘플됨) → 512 grid 실효 spacing 2배
# 원본 1024 매트릭스 (전처리로 512 다운샘플됨) → 512 grid 실효 spacing 2배.
# 해당 환자 목록은 배포하지 않습니다. configs/matrix1024.csv (column: patient_id)
MATRIX_1024 = set()
_mx_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "matrix1024.csv")
if os.path.exists(_mx_path):
    with open(_mx_path) as _f:
        for _r in _csv.DictReader(_f):
            MATRIX_1024.add(_r["patient_id"])

# 간경변 여부 라벨도 배포하지 않습니다. configs/cirrhosis.csv (columns: patient_id,cirrhosis[yes/no])
CIRRHOSIS_NO = set()
_ci_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "cirrhosis.csv")
if os.path.exists(_ci_path):
    with open(_ci_path) as _f:
        for _r in _csv.DictReader(_f):
            if _r["cirrhosis"].strip().lower()=="no": CIRRHOSIS_NO.add(_r["patient_id"])
def cirrhosis(pid):
    return "no" if pid in CIRRHOSIS_NO else "yes"


def load_isp(path):
    """patient_id → {idx: cc|None}. case ID는 .0/공백 정규화."""
    isp = {}
    if not os.path.exists(path):
        print(f"[WARN] ISP CSV 없음: {path}"); return isp
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid = str(row["case"]).strip().split(".")[0]
            vols = {}
            for col, idx in ISP_TO_CH.items():
                val = (row.get(col, "") or "").strip().strip('"')
                vols[idx] = float(val) if val else None
            isp[pid] = vols
    return isp


def load_mask(pid):
    p = os.path.join(DATA_DIR, pid, "mask.npy")
    return np.load(p) if os.path.exists(p) else None


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"저장: {path} ({len(rows)}행)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    isp_all = load_isp(ISP_CSV)
    ids = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)))
    print(f"ISP 로드: {len(isp_all)}명 | combined: {len(ids)}명")

    rows_cmp, rows_cal, rows_dis = [], [], []

    for pid in ids:
        mask = load_mask(pid)
        if mask is None:
            print(f"[{pid}] 마스크 없음 — skip"); continue

        isp = isp_all.get(pid, {})
        isp_total = sum(v for v in isp.values() if v is not None)

        # GT 복셀 (9분절: ch1~9, ch0=배경 제외)
        gt_vox = [int(mask[c].sum()) for c in range(1, N_SEG + 1)]
        gt_total_vox = sum(gt_vox)

        # 실측 spacing
        xy = REAL_SPACING.get(pid)
        src = "measured"
        # 원본 1024 → 512 다운샘플: 실효 in-plane spacing 2배 보정
        if xy is not None and pid in MATRIX_1024:
            xy *= 2.0
            src = "measured_x2(orig1024)"
        # 실측값 없으면 ISP 역산 fallback
        if xy is None and isp_total > 0 and gt_total_vox > 0:
            xy = math.sqrt(isp_total * 1000.0 / (gt_total_vox * SLICE_Z))
            src = "isp_backcalc(FALLBACK)"
            print(f"  [WARN] {pid}: 실측 spacing 없음 → ISP 역산")
        if xy is None:
            print(f"  [WARN] {pid}: spacing 불가 — skip"); continue

        vox_mm3     = xy ** 2 * SLICE_Z
        gt_total_cc = gt_total_vox * vox_mm3 / 1000.0
        gt_isp_pct  = (gt_total_cc - isp_total) / isp_total * 100 if isp_total > 0 else None

        rows_cal.append({
            "patient_id":      pid,
            "cirrhosis":       cirrhosis(pid),
            "mask_h":          mask.shape[2],
            "mask_slices":     mask.shape[1],
            "spacing_src":     src,
            "xy_spacing_mm":   round(xy, 6),
            "voxel_mm3":       round(vox_mm3, 4),
            "gt_total_voxels": gt_total_vox,
            "gt_total_cc":     round(gt_total_cc, 1),
            "isp_total_cc":    round(isp_total, 1),
            "gt_isp_diff_pct": round(gt_isp_pct, 1) if gt_isp_pct is not None else "",
        })

        # 분절별 GT(실측) vs ISP   (gt_vox[c]=ch(c+1), isp.get(c), SEG_NAMES[c] 정렬)
        for c in range(N_SEG):
            isp_v = isp.get(c)
            gt_cc = round(gt_vox[c] * vox_mm3 / 1000.0, 2)
            err_cc  = round(gt_cc - isp_v, 2)             if isp_v else ""
            err_pct = round((gt_cc - isp_v)/isp_v*100, 1) if isp_v else ""
            rows_cmp.append({
                "patient_id": pid, "segment": SEG_NAMES[c],
                "gt_cc": gt_cc, "isp_cc": isp_v,
                "err_cc": err_cc, "err_pct": err_pct,
            })

        # 병변↔분절 겹침 (GT) — 분절은 ch1~9, 병변은 ch10~17
        if mask.shape[0] >= 11:
            for les_ch, les_name in LESION_CH.items():
                if les_ch >= mask.shape[0]: continue
                les_mask = mask[les_ch]
                if les_mask.sum() == 0: continue
                for c in range(N_SEG):
                    seg_ch  = c + 1
                    overlap = int((mask[seg_ch] * les_mask).sum())
                    if overlap == 0: continue
                    rows_dis.append({
                        "patient_id":     pid,
                        "lesion":         les_name,
                        "segment":        SEG_NAMES[c],
                        "overlap_voxels": overlap,
                        "seg_gt_cc":      round(gt_vox[c] * vox_mm3 / 1000.0, 2),
                        "lesion_gt_cc":   round(int(les_mask.sum()) * vox_mm3 / 1000.0, 2),
                    })

        print(f"[{pid}] xy={xy:.4f}mm | GT={gt_total_cc:.1f}cc | ISP={isp_total:.1f}cc | "
              f"H={mask.shape[2]} sl={mask.shape[1]} | {src} | cirr={cirrhosis(pid)}")

    # ===== 저장 =====
    write_csv(f"{OUT_DIR}/volume_comparison.csv", rows_cmp,
              ["patient_id","segment","gt_cc","isp_cc","err_cc","err_pct"])
    write_csv(f"{OUT_DIR}/spacing_calibration.csv", rows_cal,
              ["patient_id","cirrhosis","mask_h","mask_slices","spacing_src","xy_spacing_mm",
               "voxel_mm3","gt_total_voxels","gt_total_cc","isp_total_cc","gt_isp_diff_pct"])
    write_csv(f"{OUT_DIR}/disease_volume.csv", rows_dis,
              ["patient_id","lesion","segment","overlap_voxels","seg_gt_cc","lesion_gt_cc"])

    # ===== 분절별 GT 기술통계 =====
    seg_gt = {n: [] for n in SEG_NAMES}
    for r in rows_cmp:
        seg_gt[r["segment"]].append(r["gt_cc"])
    rows_stat = []
    for seg in SEG_NAMES:
        a = np.array(seg_gt[seg])
        if not len(a): continue
        rows_stat.append({
            "segment": seg, "n": len(a),
            "mean_cc": round(float(a.mean()),1), "std_cc": round(float(a.std()),1),
            "min_cc": round(float(a.min()),1), "max_cc": round(float(a.max()),1),
            "median_cc": round(float(np.median(a)),1),
        })
    write_csv(f"{OUT_DIR}/segment_stats.csv", rows_stat,
              ["segment","n","mean_cc","std_cc","min_cc","max_cc","median_cc"])

    # pid → {seg: gt_cc}
    pid_gt = {}
    for r in rows_cmp:
        pid_gt.setdefault(r["patient_id"], {})[r["segment"]] = r["gt_cc"]

    # ===== 우/좌/미상엽 비율 (GT; 정상값 출처는 Abdalla 2004 후보, 원문 미대조) =====
    rows_lobe = []
    for pid, sv in pid_gt.items():
        def lobe_sum(segs):
            v = [sv[s] for s in segs if sv.get(s) is not None]
            return round(sum(v), 1) if v else None
        r_cc, l_cc, c_cc = lobe_sum(RIGHT), lobe_sum(LEFT), lobe_sum(CAUD)
        total = round((r_cc or 0)+(l_cc or 0)+(c_cc or 0), 1) or None
        rows_lobe.append({
            "patient_id": pid, "cirrhosis": cirrhosis(pid),
            "right_cc": r_cc, "left_cc": l_cc, "caudate_cc": c_cc, "total_cc": total,
            "right_pct": round(r_cc/total*100,1) if (r_cc and total) else None,
            "left_pct":  round(l_cc/total*100,1) if (l_cc and total) else None,
            "caudate_pct": round(c_cc/total*100,1) if (c_cc and total) else None,
            "right_left_ratio": round(r_cc/l_cc,2) if (r_cc and l_cc) else None,
            "ref": "Vauthey JN, Ann Surg 2000",
        })
    write_csv(f"{OUT_DIR}/lobe_ratio.csv", rows_lobe,
              ["patient_id","cirrhosis","right_cc","left_cc","caudate_cc","total_cc",
               "right_pct","left_pct","caudate_pct","right_left_ratio","ref"])

    # ===== 병변 분절 위축 (GT, Ribero 2007) =====
    LESION_TYPES = ["tumor","tace_site","ablation","lipiodol"]
    rows_atr = []
    pid_les = {}
    for r in rows_dis:
        pid_les.setdefault(r["patient_id"], []).append(r)
    for pid, les_rows in pid_les.items():
        sv = pid_gt.get(pid, {})
        total = sum(v for v in sv.values() if v) or None
        if not total: continue
        affected = {r["segment"] for r in les_rows if r["lesion"] in LESION_TYPES}
        unaff = [sv[s] for s in SEG_NAMES if s not in affected and sv.get(s)]
        unaff_mean = float(np.mean(unaff)) if unaff else None
        for r in les_rows:
            if r["lesion"] not in LESION_TYPES: continue
            seg = r["segment"]; gv = sv.get(seg)
            if not gv: continue
            lobe = "right" if seg in RIGHT else ("left" if seg in LEFT else "caudate")
            rows_atr.append({
                "patient_id": pid, "lesion_type": r["lesion"],
                "affected_segment": seg, "lobe": lobe,
                "affected_vol_cc": gv, "affected_vol_pct": round(gv/total*100,1),
                "unaffected_mean_cc": round(unaff_mean,1) if unaff_mean else None,
                "ratio_vs_unaffected": round(gv/unaff_mean,2) if unaff_mean else None,
                "ref": "Ribero D, Ann Surg 2007; Chapiro J, Radiology 2014",
            })
    write_csv(f"{OUT_DIR}/lesion_atrophy.csv", rows_atr,
              ["patient_id","lesion_type","affected_segment","lobe",
               "affected_vol_cc","affected_vol_pct","unaffected_mean_cc",
               "ratio_vs_unaffected","ref"])

    # ===== FLR (GT; 20% 기준 원출처=Kishi 2009, 원문확인) =====
    rows_flr = []
    for pid, sv in pid_gt.items():
        def ss(segs):
            v = [sv[s] for s in segs if sv.get(s) is not None]
            return round(sum(v),1) if v else None
        total = ss(SEG_NAMES)
        rhr = ss(["Seg1","Seg2","Seg3","Seg4a","Seg4b"])
        lhr = ss(["Seg1","Seg5","Seg6","Seg7","Seg8"])
        thr = 0.40 if cirrhosis(pid) == "yes" else 0.20
        rows_flr.append({
            "patient_id": pid, "cirrhosis": cirrhosis(pid), "total_cc": total,
            "flr_right_hep_cc": rhr,
            "flr_right_hep_pct": round(rhr/total*100,1) if (rhr and total) else None,
            "flr_left_hep_cc": lhr,
            "flr_left_hep_pct": round(lhr/total*100,1) if (lhr and total) else None,
            "safe_thr_pct": int(thr*100),
            "safe_right_hep": "YES" if (rhr and total and rhr/total > thr) else "BORDERLINE",
            "safe_left_hep":  "YES" if (lhr and total and lhr/total > thr) else "BORDERLINE",
            "ref": "Kishi Y, Ann Surg 2009",
        })
    write_csv(f"{OUT_DIR}/flr.csv", rows_flr,
              ["patient_id","cirrhosis","total_cc","flr_right_hep_cc","flr_right_hep_pct",
               "flr_left_hep_cc","flr_left_hep_pct","safe_thr_pct",
               "safe_right_hep","safe_left_hep","ref"])

    # ===== 콘솔 요약 =====
    print("\n===== GT vs ISP 총량 일치도 (실측 spacing) =====")
    d = [r["gt_isp_diff_pct"] for r in rows_cal if isinstance(r["gt_isp_diff_pct"], (int,float))]
    if d:
        a = np.array(d)
        print(f"  N={len(a)} 평균={a.mean():+.1f}% SD={a.std():.1f}% |오차|평균={np.abs(a).mean():.1f}%")
        worst = sorted([r for r in rows_cal if isinstance(r["gt_isp_diff_pct"],(int,float))],
                       key=lambda r: abs(r["gt_isp_diff_pct"]), reverse=True)[:8]
        print("  [오차 큰 환자]")
        for r in worst:
            print(f"    {r['patient_id']} GT={r['gt_total_cc']} ISP={r['isp_total_cc']} "
                  f"diff={r['gt_isp_diff_pct']:+.1f}% (xy={r['xy_spacing_mm']}, H={r['mask_h']}, {r['spacing_src']})")

    print("\n===== 분절별 GT 부피 =====")
    for r in rows_stat:
        print(f"  {r['segment']:<8}: {r['mean_cc']:>7} ± {r['std_cc']:<6} "
              f"(min {r['min_cc']}, max {r['max_cc']}, n={r['n']})")

    rp = [r["right_pct"] for r in rows_lobe if r["right_pct"]]
    lp = [r["left_pct"]  for r in rows_lobe if r["left_pct"]]
    cp = [r["caudate_pct"] for r in rows_lobe if r["caudate_pct"]]
    if rp:
        print("\n===== 우/좌/미상엽 비율 (GT, Vauthey 2000) =====")
        print(f"  우엽:   {np.mean(rp):.1f}% ± {np.std(rp):.1f} (정상 65-70%)")
        print(f"  좌엽:   {np.mean(lp):.1f}% ± {np.std(lp):.1f} (정상 25-30%)")
        print(f"  미상엽: {np.mean(cp):.1f}% ± {np.std(cp):.1f} (정상 ~5%)")


if __name__ == "__main__":
    main()