#!/usr/bin/env python
"""병변-분절 위치 판정 ROC 분석 (서버측) — 센터들의 lesion_overlap.csv 를 모아
"병변이 분절 s에 있다"(gt_frac ≥ τ_gt)를 정답으로, pred_frac 을 점수로 임계치 스윕한다.

  python scripts/analyze_lesion_roc.py --roots outputs/collected/exp4c/<run> outputs/exp4c/<run> [--tau-gt 0.05]
출력: outputs/analysis_lesion/<시각>/ (실행마다 새 폴더)
  - lesion_pairs.csv        병변×분절 전체 표(site/fold/method 포함)
  - roc_by_method.csv/json  방법론별 AUC·최적 임계치(Youden)·민감도/특이도/정확도
  - roc_by_site.csv         센터별(방법론 고정 시)
  - roc_curves.png          방법론별 ROC 곡선
"""
import os, sys, glob, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from couinaudfl.data import SEG_NAMES


def collect(roots):
    rows = []
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "lesion_overlap.csv"), recursive=True):
            rel = os.path.relpath(f, root); parts = rel.replace("\\", "/").split("/")
            site = next((p.split("client_")[-1].split("_")[-1] for p in parts if "client_" in p or p in list("ABCDE")), "?")
            for p in parts:
                if p.startswith("client_"): site = p[len("client_"):]
            fold = next((int(p[4:]) for p in parts if p.startswith("fold") and p[4:].isdigit()), -1)
            method = next((p for p in parts if p in ("FedAvg", "FedProx", "FedAdam", "FedBN", "Single")), parts[-2] if len(parts) > 1 else "?")
            df = pd.read_csv(f); df.insert(0, "method", method); df.insert(0, "fold", fold); df.insert(0, "site", site)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else None


def to_pairs(df, tau_gt):
    recs = []
    for _, r in df.iterrows():
        for n in SEG_NAMES:
            recs.append({"site": r.site, "fold": r.fold, "method": r.method, "case": r.case, "lesion_id": r.lesion_id,
                         "lesion_ml": r.lesion_ml, "segment": n, "present": int(r[f"gt_frac_{n}"] >= tau_gt), "score": float(r[f"pred_frac_{n}"])})
    return pd.DataFrame(recs)


def roc(pairs):
    # 판정 규칙: score > thr (thr=0 → 겹침이 조금이라도 있으면 양성). 앵커: thr<0 전부 양성(1,1), thr≥1 전부 음성(0,0).
    ths = np.concatenate([[-1.0], np.round(np.linspace(0, 1, 201), 3)])
    y = (pairs.present == 1).values; sc = pairs.score.values
    P = int(y.sum()); N = int((~y).sum()); pts = []
    for t in ths:
        pred = sc > t
        tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
        pts.append({"thr": float(t), "tpr": tp / max(P, 1), "fpr": fp / max(N, 1)})
    d = pd.DataFrame(pts).sort_values(["fpr", "tpr"])
    auc = float(np.trapz(d.tpr.values, d.fpr.values))
    di = d[d.thr >= 0].copy(); di["youden"] = di.tpr - di.fpr; best = di.loc[di.youden.idxmax()]
    pred = sc > best.thr; acc = float((pred == y).mean())
    return d, {"auc": round(auc, 4), "thr_opt": float(best.thr), "sens": round(float(best.tpr), 4),
               "spec": round(float(1 - best.fpr), 4), "acc": round(acc, 4), "n_pos": P, "n_neg": N}


def load_type_maps(roots):
    maps = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "lesion_labels.json"), recursive=True):
            site = next((p[len("client_"):] for p in f.split(os.sep) if p.startswith("client_")), "?")
            try:
                m = {int(k): v for k, v in json.load(open(f, encoding="utf-8")).items() if k.isdigit() and str(v).strip()}
                if m: maps[site] = m
            except Exception: pass
    return maps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--tau-gt", type=float, default=0.05)
    ap.add_argument("--tau-gt-sweep", type=float, nargs="*", default=[0.01, 0.05, 0.1, 0.25, 0.5], help="GT 겹침 임계치 민감도 분석")
    ap.add_argument("--out", default=None); a = ap.parse_args()
    out = a.out or os.path.join("outputs", "analysis_lesion", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")); os.makedirs(out, exist_ok=True)
    df = collect(a.roots)
    if df is None: print("lesion_overlap.csv 없음"); sys.exit(1)
    tmap = load_type_maps(a.roots)
    df["lesion_type"] = [ (tmap.get(r.site, {}) or {}).get(int(r.lesion_channel), f"ch{int(r.lesion_channel)}") for r in df.itertuples() ]
    df.to_csv(os.path.join(out, "lesions_all.csv"), index=False)
    # fold 구성표: 센터×fold×유형별 병변 수·총 부피 (GT는 방법론과 무관 → 방법론 하나로 dedup)
    one = df.drop_duplicates(subset=["site", "fold", "case", "lesion_id"])
    comp = one.groupby(["site", "fold", "lesion_type"]).agg(n_lesions=("lesion_id", "count"), total_ml=("lesion_ml", "sum"), n_cases=("case", "nunique")).round(2)
    comp.to_csv(os.path.join(out, "fold_composition.csv")); print("fold 구성:\n", comp.head(20))
    pairs = to_pairs(df, a.tau_gt); pairs["lesion_type"] = np.repeat(df["lesion_type"].values, len(SEG_NAMES))
    pairs.to_csv(os.path.join(out, "lesion_pairs.csv"), index=False)
    res = {}; curves = {}
    for m, g in pairs.groupby("method"):
        curves[m], res[m] = roc(g)
    pd.DataFrame(res).T.to_csv(os.path.join(out, "roc_by_method.csv")); json.dump(res, open(os.path.join(out, "roc_by_method.json"), "w"), indent=1)
    # 교차 fold 작동점: fold k 평가용 θ는 나머지 fold의 Youden으로 선택(임계치 선택 편향 제거)
    cv_rows = []
    for m, g in pairs.groupby("method"):
        folds = sorted(g.fold.unique())
        if len(folds) < 2: continue
        tp = fp = fn = tn = 0; thr_used = []
        for k in folds:
            tr = g[g.fold != k]; te = g[g.fold == k]
            _, r = roc(tr); th = r["thr_opt"]; thr_used.append(th)
            pred = te.score.values > th; y = (te.present == 1).values
            tp += int((pred & y).sum()); fp += int((pred & ~y).sum()); fn += int((~pred & y).sum()); tn += int((~pred & ~y).sum())
        cv_rows.append({"method": m, "sens_cv": round(tp / max(tp + fn, 1), 4), "spec_cv": round(tn / max(tn + fp, 1), 4),
                        "acc_cv": round((tp + tn) / max(tp + fp + fn + tn, 1), 4), "thr_range": f"{min(thr_used):.3f}-{max(thr_used):.3f}", "n_folds": len(folds)})
    if cv_rows: pd.DataFrame(cv_rows).to_csv(os.path.join(out, "operating_point_cv.csv"), index=False); print("교차 fold 작동점:\n", pd.DataFrame(cv_rows))
    site_rows = []
    for (m, s), g in pairs.groupby(["method", "site"]):
        _, r = roc(g); r.update({"method": m, "site": s}); site_rows.append(r)
    pd.DataFrame(site_rows).to_csv(os.path.join(out, "roc_by_site.csv"), index=False)
    if "com_gt_seg" in df.columns:   # 중심 기준 단일 분절 일치율(간 내 중심만)
        c = df[(df.com_gt_seg > 0)]
        com = c.groupby("method").apply(lambda g: pd.Series({"n": len(g), "com_acc": round(float((g.com_gt_seg == g.com_pred_seg).mean()), 4)}), include_groups=False)
        com.to_csv(os.path.join(out, "centroid_accuracy.csv")); print("중심 기준 일치율:\n", com)
    type_rows = []
    for (m, t), g in pairs.groupby(["method", "lesion_type"]):
        if (g.present == 1).sum() >= 10:
            _, r = roc(g); r.update({"method": m, "lesion_type": t, "n_lesions": int(g.drop_duplicates(["site","fold","case","lesion_id"]).shape[0])}); type_rows.append(r)
    pd.DataFrame(type_rows).to_csv(os.path.join(out, "roc_by_type.csv"), index=False)
    sweep_rows = []
    for tg in a.tau_gt_sweep:
        pw = to_pairs(df, tg)
        for m, g in pw.groupby("method"):
            _, r = roc(g); sweep_rows.append({"tau_gt": tg, "method": m, **r})
    pd.DataFrame(sweep_rows).to_csv(os.path.join(out, "tau_gt_sweep.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    for m, d in curves.items(): plt.plot(d.fpr, d.tpr, label=f"{m} (AUC {res[m]['auc']:.3f}, θ* {res[m]['thr_opt']:.2f})")
    plt.plot([0, 1], [0, 1], "k:", lw=0.8); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=8)
    plt.title(f"Lesion-in-segment ROC (τ_gt={a.tau_gt})"); plt.tight_layout(); plt.savefig(os.path.join(out, "roc_curves.png"), dpi=140)
    print(json.dumps(res, indent=1)); print("→", out)


if __name__ == "__main__":
    main()
