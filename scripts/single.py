#!/usr/bin/env python
"""단일센터 5-fold — FL과 동일 분할·초기화·에폭(rounds×local_epochs)·설정으로 로컬만 학습.

  python scripts/single.py --config configs/exp4c.yaml --data D:/data/liver --site A [--folds 0 1]
결과: outputs/<exp>/client_<site>/fold<k>/Single/{best.pth,last.pth,test_metrics.csv,test_summary.json}
"""
import os, sys, json, argparse, datetime, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, pandas as pd, signal
for _sig in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, _sig): signal.signal(getattr(signal, _sig), signal.SIG_IGN)
from couinaudfl.model import build_model
from couinaudfl.data import list_cases, kfold_split, load_case
from couinaudfl.train import fit
from couinaudfl.infer import predict_volume
from couinaudfl.metrics import case_metrics, summarize

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--data", required=True); ap.add_argument("--site", default=socket.gethostname())

def main():
    ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--run", default=None, help="실행 이름(기본: 현재 시각). 같은 이름으로 재실행하면 이어서 함"); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    C = yaml.safe_load(open(a.config)); folds = a.folds if a.folds is not None else C.get("folds", [0, 1, 2, 3, 4])
    run = a.run or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"); root = os.path.join(C.get("output_dir", "outputs"), C["experiment"], run, f"client_{a.site}"); os.makedirs(root, exist_ok=True)
    logf = open(os.path.join(root, "single.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda"); cases = list_cases(a.data); epochs = C["rounds"] * C["local_epochs"]; amp = bool(C.get("amp", 0))
    for fold in folds:
        od = os.path.join(root, f"fold{fold}", "Single"); done = os.path.join(od, "DONE")
        if os.path.exists(os.path.join(root, "STOP.txt")): log("STOP.txt — 종료"); break
        if os.path.exists(done): log(f"skip fold{fold}"); continue
        tr, va, te = kfold_split(cases, fold, C.get("n_folds", 5), C.get("seed", 42), C.get("val_frac", 0.1))
        json.dump({"train": [os.path.basename(x) for x in tr], "val": [os.path.basename(x) for x in va], "test": [os.path.basename(x) for x in te]},
                  open(os.path.join(root, f"fold{fold}_split.json"), "w"), indent=1)
        model = build_model().to(dev)
        if C.get("init_weights"): sd = torch.load(C["init_weights"], map_location="cpu", weights_only=False); model.load_state_dict(sd.get("model", sd))
        log(f"=== Single fold {fold}: train {len(tr)} val {len(va)} test {len(te)} epochs {epochs}")
        r = fit(model, tr, va, od, dev, epochs=epochs, lr=C.get("lr", 0.01), amp=amp, num_workers=a.workers, val_every=C.get("local_epochs", 10), log=log)
        model.load_state_dict(torch.load(os.path.join(od, "best.pth"), map_location=dev)); model.eval(); rows = []
        with torch.no_grad():
            for cd in te:
                img, lab, meta = load_case(cd); pred, _ = predict_volume(model, img, dev, amp=amp); sp = meta.get("spacing") or [4.0, 0.7, 0.7]
                rows += case_metrics(pred, lab, [4.0, float(sp[1]), float(sp[2])], os.path.basename(cd))
                os.makedirs(os.path.join(od, "pred"), exist_ok=True); import numpy as np
                np.savez_compressed(os.path.join(od, "pred", f"{os.path.basename(cd)}.npz"), pred=pred)
        pd.DataFrame(rows).to_csv(os.path.join(od, "test_metrics.csv"), index=False); s = summarize(rows)
        json.dump(s, open(os.path.join(od, "test_summary.json"), "w"), indent=1); log(f"fold {fold} Single test {json.dumps(s)}")
        open(done, "w").write(str(datetime.datetime.now()))
    log("단일센터 완료")


if __name__ == "__main__":
    main()