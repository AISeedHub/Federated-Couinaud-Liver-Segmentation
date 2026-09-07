#!/usr/bin/env python
"""G-UNETR++ (Lee 2026, Curr Probl Surg) 공개 체크포인트 zero-shot 평가.

  python scripts/eval_gunetr_zero_shot.py --cases /home/dspserver/2025/jin/combined_80 [--liver-mask gt]
전처리(그들 MSD_npy_make.py 재현): 1mm 등방 리샘플 → 전체 영상 min-max [0,1] → 간 마스킹(밖=0)
→ 슬라이딩 윈도우 64×128×128 → argmax → 원 격자 복원(최근접). 8분절(4a/4b 병합) 기준. uint8 min-max는 HU min-max와 동일.
--liver-mask gt : 원 논문의 캐스케이드(전체 간 마스크 후 분절)를 GT 간(분절 합집합)으로 대신함 = 경쟁 모델에 유리한(oracle) 조건. 논문 표에 명시.
"""
import os, sys, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, pandas as pd
from scipy.ndimage import zoom
from monai.inferers import sliding_window_inference
from couinaudfl.data import load_case, list_cases
from couinaudfl.metrics import case_metrics, summarize

TGT_SP = np.array([1.0, 1.0, 1.0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", required=True); ap.add_argument("--list", default=None); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repo", default="/data/campaign/gunetr_repo"); ap.add_argument("--ckpt", default="/data/datasets/couinaud_public/checkpoints/gunetr_pp_couinaud.zip")
    ap.add_argument("--liver-mask", choices=["none", "gt"], default="gt"); ap.add_argument("--out", default=None); ap.add_argument("--no-hd", action="store_true")
    a = ap.parse_args(); sys.path.insert(0, a.repo)
    from unetr_pp.network_architecture.synapse.unetr_pp_synapse import UNETR_PP
    dev = torch.device("cuda")
    model = UNETR_PP(in_channels=1, out_channels=9, img_size=[64, 128, 128], feature_size=16, num_heads=4, depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=False).to(dev)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); miss, unexp = model.load_state_dict(ck["state_dict"], strict=False); model.eval()
    cases = list_cases(a.cases)
    if a.list:
        jp, key = a.list.split(":"); sel = set(os.path.basename(x) for x in json.load(open(jp))[key]); cases = [c for c in cases if os.path.basename(c) in sel]
    if a.limit: cases = cases[:a.limit]
    out = a.out or f"outputs/zeroshot_gunetr_{os.path.basename(a.cases.rstrip('/'))}_{a.liver_mask}"; os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    log(f"G-UNETR++ zero-shot on {a.cases}: {len(cases)} cases | ckpt missing {len(miss)} unexpected {len(unexp)} | liver-mask {a.liver_mask}")
    rows = []
    with torch.no_grad():
        for i, cd in enumerate(cases):
            name = os.path.basename(cd); img, lab, meta = load_case(cd)
            sp = meta.get("spacing") or meta.get("orig_spacing") or [4.0, 0.73, 0.73]; src = np.array([4.0, float(sp[1]), float(sp[2])])
            pred_p = os.path.join(out, f"pred_{name}.npy")
            if os.path.exists(pred_p): pred = np.load(pred_p)
            else:
                v = img.astype(np.float32); v = (v - v.min()) / max(v.max() - v.min(), 1e-6); f = src / TGT_SP; vr = zoom(v, f, order=1)
                if a.liver_mask == "gt":
                    lm = zoom((lab > 0).astype(np.uint8), f, order=0).astype(bool); vr = np.where(lm, vr, 0.0)
                t = torch.from_numpy(vr[None, None].astype(np.float32)).to(dev)
                with torch.autocast("cuda"):
                    lo = sliding_window_inference(t, (64, 128, 128), 2, model, overlap=0.5, mode="gaussian")
                p8 = lo.argmax(1).cpu().numpy()[0].astype(np.uint8); po = zoom(p8, 1 / f, order=0)
                pred8 = np.zeros(img.shape, np.uint8); sz = [min(x1, x2) for x1, x2 in zip(po.shape, img.shape)]
                pred8[:sz[0], :sz[1], :sz[2]] = po[:sz[0], :sz[1], :sz[2]]
                if a.liver_mask == "gt": pred8[lab == 0] = 0
                pred = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9], np.uint8)[pred8]; np.save(pred_p, pred)   # 8분절 → 10클래스 규약
            rows += case_metrics(pred, lab, list(src), name, seg8=True, with_hd=not a.no_hd)
            if (i + 1) % 10 == 0: log(f"  {i+1}/{len(cases)} 누적 분절 Dice {summarize(rows)['dice_mean_segments']:.4f}")
    pd.DataFrame(rows).to_csv(os.path.join(out, "metrics.csv"), index=False); s = summarize(rows)
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()
