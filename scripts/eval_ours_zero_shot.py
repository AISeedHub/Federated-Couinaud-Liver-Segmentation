#!/usr/bin/env python
"""우리 가중치(사전학습/FL 글로벌)를 기관 포맷 데이터에 zero-shot 평가. 8분절(4a/4b 병합)과 9분절 둘 다 기록.
  python scripts/eval_ours_zero_shot.py --weights outputs/pretrain/best.pth --cases /home/dspserver/2025/jin/combined_80 --out outputs/zeroshot/pretrain_bundang
"""
import os, sys, json, argparse, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, pandas as pd
from couinaudfl.model import build_model
from couinaudfl.data import load_case, list_cases
from couinaudfl.infer import predict_volume
from couinaudfl.metrics import case_metrics, summarize


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--cases", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--no-hd", action="store_true"); ap.add_argument("--amp", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, "eval.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda"); m = build_model().to(dev); sd = torch.load(a.weights, map_location=dev, weights_only=False); m.load_state_dict(sd.get("model", sd)); m.eval()
    cases = list_cases(a.cases); cases = cases[:a.limit] if a.limit else cases; log(f"{a.weights} → {a.cases}: {len(cases)} cases")
    r8, r9 = [], []
    for i, cd in enumerate(cases):
        img, lab, meta = load_case(cd); name = os.path.basename(cd); sp = meta["spacing"]
        pp = os.path.join(a.out, f"pred_{name}.npy")
        if os.path.exists(pp): pred = np.load(pp)
        else: pred, _ = predict_volume(m, img, dev, amp=bool(a.amp)); np.save(pp, pred)
        r8 += case_metrics(pred, lab, sp, name, seg8=True, with_hd=not a.no_hd); r9 += case_metrics(pred, lab, sp, name, seg8=False, with_hd=False)
        if (i + 1) % 10 == 0: log(f"  {i+1}/{len(cases)} 누적 8분절 Dice {summarize(r8)['dice_mean_segments']:.4f}")
    pd.DataFrame(r8).to_csv(os.path.join(a.out, "metrics_8seg.csv"), index=False); pd.DataFrame(r9).to_csv(os.path.join(a.out, "metrics_9seg.csv"), index=False)
    s = {"seg8": summarize(r8), "seg9": summarize(r9)}; json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1); log(json.dumps(s))


if __name__ == "__main__":
    main()
