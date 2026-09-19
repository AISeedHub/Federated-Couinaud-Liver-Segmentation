# -*- coding: utf-8 -*-
"""CRLM FLR — 비교군(G-UNETR++/TS570) 평가. 예측은 각 원저자 공식 파이프라인, FLR 채점은 eval_crlm_flr.py와 동일 로직.
사용:
  .venv/bin/python scripts/eval_crlm_flr_baselines.py --model gunetr --out outputs/zeroshot/crlm_flr_gunetr
  .venv/bin/python scripts/eval_crlm_flr_baselines.py --model ts570  --out outputs/zeroshot/crlm_flr_ts570
CRLM GT: mask.npy ch1..5 = liver/remnant/hv/pv/tumor → 간 = ch1|2|5, 잔존 = ch2.
G-UNETR++는 원 레시피(1mm iso, minmax, 간 마스크 캐스케이드)를 GT 간(oracle)으로 대신함 — 표에 각주.
TS570은 TotalSegmentator 공식 CLI(liver_segments)로 NIfTI 왕복.
"""
from __future__ import annotations
import os, sys, json, argparse, datetime
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from couinaudfl.data import list_cases
from couinaudfl.metrics import dice as _dice, hd95
from couinaudfl.data import SEG_NAMES

TGT_SP = np.array([1.0, 1.0, 1.0])


def load_crlm(cd):
    from couinaudfl.data import load_case
    img, lab, meta = load_case(cd)          # lab = argmax(ch), CRLM에선 1..5 = liver/remnant/hv/pv/tumor
    sp = meta["spacing"]
    gt_liver = np.isin(lab, [1, 2, 5]); gt_flr = lab == 2   # eval_crlm_flr.py와 동일 규약
    return img, gt_liver, gt_flr, [float(sp[0]), float(sp[1]), float(sp[2])]


def flr_rows(name, pred, gt_liver, gt_flr, sp):
    vox = float(np.prod(sp)) / 1000.0
    pr_liver = pred > 0
    sel = []
    for c in range(1, 10):
        pc = pred == c
        if pc.sum() > 0 and (pc & gt_flr).sum() / pc.sum() > 0.5: sel.append(c)
    pr_flr = np.isin(pred, sel)
    return {"case": name, "liver_dice": _dice(pr_liver, gt_liver),
            "liver_vol_gt_ml": float(gt_liver.sum() * vox), "liver_vol_pred_ml": float(pr_liver.sum() * vox),
            "flr_dice": _dice(pr_flr, gt_flr) if sel else 0.0,
            "flr_vol_gt_ml": float(gt_flr.sum() * vox), "flr_vol_pred_ml": float(pr_flr.sum() * vox),
            "flr_hd95_mm": hd95(pr_flr, gt_flr, sp) if sel else float("nan"),
            "flr_pct_gt": float(gt_flr.sum() / max(gt_liver.sum(), 1) * 100),
            "flr_pct_pred": float(pr_flr.sum() / max(pr_liver.sum(), 1) * 100),
            "segments_selected": "+".join(SEG_NAMES[c - 1] for c in sel)}


def predict_gunetr(cases, out, repo, ckpt, log):
    import torch
    from scipy.ndimage import zoom
    from monai.inferers import sliding_window_inference
    sys.path.insert(0, repo)
    from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
    dev = torch.device("cuda")
    model = UNETR_PP(in_channels=1, out_channels=9, img_size=[64, 128, 128], feature_size=16, num_heads=4,
                     depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=False).to(dev)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=False); model.eval()
    LUT = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9], np.uint8)
    for i, cd in enumerate(cases):
        name = os.path.basename(cd); pp = os.path.join(out, f"pred_{name}.npy")
        if os.path.exists(pp): continue
        img, gt_liver, _, sp = load_crlm(cd)
        src = np.array([sp[0], sp[1], sp[2]]); f = src / TGT_SP
        v = img.astype(np.float32); v = (v - v.min()) / max(v.max() - v.min(), 1e-6)
        vr = zoom(v, f, order=1)
        lm = zoom(gt_liver.astype(np.uint8), f, order=0).astype(bool); vr = np.where(lm, vr, 0.0)
        import torch as _t
        t = _t.from_numpy(vr[None, None].astype(np.float32)).to(dev)
        with _t.no_grad(), _t.autocast("cuda"):
            lo = sliding_window_inference(t, (64, 128, 128), 1, model, overlap=0.5, mode="gaussian")
        p8 = lo.argmax(1).cpu().numpy()[0].astype(np.uint8)
        po = zoom(p8, 1 / f, order=0)
        pred8 = np.zeros(img.shape, np.uint8); sz = [min(a, b) for a, b in zip(po.shape, img.shape)]
        pred8[:sz[0], :sz[1], :sz[2]] = po[:sz[0], :sz[1], :sz[2]]
        pred8[~gt_liver] = 0
        np.save(pp, LUT[pred8])
        if (i + 1) % 10 == 0: log(f"  gunetr pred {i+1}/{len(cases)}")


def predict_ts570(cases, out, log):
    """TotalSegmentator 공식 파이프라인(liver_segments). eval_ts_zero_shot.py와 동일 왕복(export_nifti 헬퍼 재사용)."""
    import shutil, tempfile
    from couinaudfl.export_nifti import save_nifti, u8_to_hu, load_nifti_as_inst
    from totalsegmentator.python_api import totalsegmentator
    tmp_root = os.path.join(out, "_tmp"); os.makedirs(tmp_root, exist_ok=True)
    for i, cd in enumerate(cases):
        name = os.path.basename(cd); pp = os.path.join(out, f"pred_{name}.npy")
        if os.path.exists(pp): continue
        img, _, _, sp = load_crlm(cd)
        td = tempfile.mkdtemp(dir=tmp_root)
        inp = os.path.join(td, "in.nii.gz"); outp = os.path.join(td, "seg.nii.gz")
        save_nifti(u8_to_hu(img), sp, inp)
        totalsegmentator(inp, outp, task="liver_segments", ml=True, quiet=True, fast=False,
                         nr_thr_resamp=1, nr_thr_saving=1, device="gpu")
        seg = load_nifti_as_inst(outp)
        pred = np.zeros(img.shape, np.uint8)
        for k, v in zip(range(1, 9), [1, 2, 3, 4, 6, 7, 8, 9]): pred[seg == k] = v
        np.save(pp, pred); shutil.rmtree(td, ignore_errors=True)
        if (i + 1) % 5 == 0: log(f"  ts570 pred {i+1}/{len(cases)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gunetr", "ts570"], required=True)
    ap.add_argument("--cases", default="/data/campaign/public_v2/07_crlm")
    ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repo", default="/data/campaign/gunetr_repo")
    ap.add_argument("--ckpt", default="/data/datasets/couinaud_public/checkpoints/gunetr_pp_couinaud.zip")
    ap.add_argument("--no-hd", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    cases = list_cases(a.cases)
    cases = [c for c in cases if os.path.basename(c) != "CRLM-CT-1075"]  # 기존 제외 규약 유지
    if a.limit: cases = cases[:a.limit]
    log(f"CRLM FLR baseline={a.model}: {len(cases)} cases")
    if a.model == "gunetr": predict_gunetr(cases, a.out, a.repo, a.ckpt, log)
    else: predict_ts570(cases, a.out, log)
    rows = []
    for cd in cases:
        name = os.path.basename(cd); pp = os.path.join(a.out, f"pred_{name}.npy")
        if not os.path.exists(pp): continue
        img, gt_liver, gt_flr, sp = load_crlm(cd)
        if gt_flr.sum() == 0 or gt_liver.sum() == 0: continue
        pred = np.load(pp)
        r = flr_rows(name, pred, gt_liver, gt_flr, sp)
        if a.no_hd: r["flr_hd95_mm"] = float("nan")
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(a.out, "metrics_flr.csv"), index=False)
    s = {"n": len(df), "liver_dice_mean": float(df.liver_dice.mean()), "flr_dice_mean": float(df.flr_dice.mean()),
         "flr_pct_bias_mean": float((df.flr_pct_pred - df.flr_pct_gt).mean()),
         "flr_pct_bias_sd": float((df.flr_pct_pred - df.flr_pct_gt).std()),
         "flr_vol_corr": float(df.flr_vol_gt_ml.corr(df.flr_vol_pred_ml)),
         "liver_vol_corr": float(df.liver_vol_gt_ml.corr(df.liver_vol_pred_ml))}
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()
