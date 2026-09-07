#!/usr/bin/env python
"""FL 클라이언트 — 서버와 같은 fold × 방법론 순서로 자동 재접속. 데이터는 기관 포맷 폴더(환자별 image.npy/mask.npy).

  python scripts/client.py --config configs/exp4c.yaml --data D:/data/liver --site A
서버가 각 세션을 끝내면 클라이언트는 다음 (fold, method)로 넘어가 재접속을 시도한다(연결 실패 시 30초 간격 재시도).
STOP.txt 가 outputs/<exp>/ 에 있으면 현 세션 후 종료. 완료된 (fold, method)는 DONE 마커로 건너뜀.
"""
import os, sys, time, json, argparse, datetime, socket
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, flwr as fl, signal
# 터미널 오조작(Ctrl+C/Break) 무시 — 종료는 STOP.txt 로만
for _sig in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, _sig): signal.signal(getattr(signal, _sig), signal.SIG_IGN)
if os.name == "nt":
    try:
        import ctypes; k = ctypes.windll.kernel32; h = k.GetStdHandle(-10); m = ctypes.c_uint(); k.GetConsoleMode(h, ctypes.byref(m))
        k.SetConsoleMode(h, m.value & ~0x0040 | 0x0080)   # QuickEdit off, ExtendedFlags on
        k.SetThreadExecutionState(0x80000003)              # 절전·화면꺼짐 방지
    except Exception: pass
from couinaudfl.model import build_model
from couinaudfl.data import list_cases, kfold_split
from couinaudfl.fl import CouinaudClient, METHODS

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--data", required=True); ap.add_argument("--site", default=socket.gethostname())
ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--run", default=None, help="실행 이름(기본: 현재 시각). 같은 이름으로 재실행하면 이어서 함"); ap.add_argument("--methods", nargs="*", default=None); ap.add_argument("--workers", type=int, default=4)

def main():
    ap.add_argument("--server", default=None, help="override server_address"); a = ap.parse_args()
    C = yaml.safe_load(open(a.config)); folds = a.folds if a.folds is not None else C.get("folds", [0, 1, 2, 3, 4]); methods = a.methods or C.get("methods", METHODS)
    server = a.server or C["client_server_address"]; run = a.run or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"); root = os.path.join(C.get("output_dir", "outputs"), C["experiment"], run, f"client_{a.site}"); os.makedirs(root, exist_ok=True)
    logf = open(os.path.join(root, "client.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); cases = list_cases(a.data)
    log(f"[{C['experiment']}] site {a.site} | {len(cases)} patients | server {server} | device {dev} {torch.cuda.get_device_name(0) if dev.type=='cuda' else ''}")
    amp = bool(C.get("amp", 0))
    for fold in folds:
        tr, va, te = kfold_split(cases, fold, C.get("n_folds", 5), C.get("seed", 42), C.get("val_frac", 0.1))
        json.dump({"train": [os.path.basename(x) for x in tr], "val": [os.path.basename(x) for x in va], "test": [os.path.basename(x) for x in te]},
                  open(os.path.join(root, f"fold{fold}_split.json"), "w"), indent=1)
        for method in methods:
            od = os.path.join(root, f"fold{fold}"); done = os.path.join(od, f"DONE_{method}")
            if os.path.exists(os.path.join(root, "STOP.txt")): log("STOP.txt — 종료"); sys.exit(0)
            if os.path.exists(done): log(f"skip fold{fold} {method}"); continue
            log(f"=== fold {fold} / {method} : train {len(tr)} val {len(va)} test {len(te)} → {server}")
            model = client = None
            while True:
                try:
                    if client is None:   # 최초 또는 예외 후: 모델·클라이언트를 새로 만들어 GPU 상태를 깨끗이
                        model = build_model().to(dev)
                        client = CouinaudClient(model, tr, va, te, dev, od, amp=amp, workers=a.workers, log=log, cid=a.site)
                    fl.client.start_client(server_address=server, client=client.to_client()); break
                except Exception as e:
                    import traceback, gc; tb = traceback.format_exc()
                    if "grpc" in tb.lower() or "connect" in str(e).lower() or "UNAVAILABLE" in tb:
                        log(f"연결 대기 ({type(e).__name__}) 30s 후 재시도"); time.sleep(30)
                    else:   # 학습 중 예외 — 기록 후 모델을 버리고 새로 만들어 재접속(서버는 round_timeout 후 다음 라운드로 진행)
                        log(f"학습 예외 → {os.path.join(root, 'errors.log')} 기록, 모델 재생성 후 30s 뒤 재접속"); open(os.path.join(root, "errors.log"), "a").write(f"\n{datetime.datetime.now()} fold{fold} {method}\n{tb}")
                        del client, model; client = model = None; gc.collect(); torch.cuda.empty_cache(); time.sleep(30)
            open(done, "w").write(str(datetime.datetime.now())); log(f"fold {fold} {method} 완료")
            del model, client; torch.cuda.empty_cache(); time.sleep(C.get("gap_sec", 10))
    log("모든 세션 완료")


if __name__ == "__main__":
    main()