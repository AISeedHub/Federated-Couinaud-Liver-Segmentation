#!/usr/bin/env python
"""FL 서버 — 설정 yaml을 여러 개 주면 순서대로(4센터 exp4c 완료 후 5센터 exp5c) 실행. 실험마다 포트가 다르다(9595/9596).

  python scripts/server.py --config configs/exp4c.yaml configs/exp5c.yaml --run run_main   # 한 번의 실행으로 4센터→5센터 + 결과 수집기(9598)
  python scripts/server.py --config configs/exp4c.yaml [--folds 0 1 2 3 4] [--methods FedAvg FedProx FedAdam FedBN]
서버는 각 (fold, method) 세션마다 새 Flower 서버를 띄우고, 초기 파라미터는 사전학습 가중치(init_weights)로 준다.
"""
import os, sys, json, argparse, time, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml, torch, flwr as fl
from flwr.common import ndarrays_to_parameters
from couinaudfl.model import build_model
from couinaudfl.fl import CouinaudStrategy, get_nd, METHODS

ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True, nargs="+", help="여러 개 주면 순서대로 실행(예: exp4c.yaml exp5c.yaml)"); ap.add_argument("--folds", type=int, nargs="*", default=None); ap.add_argument("--collect-port", type=int, default=9598, help="결과 수집 서버 포트(0=끄기)"); ap.add_argument("--run", default=None, help="실행 이름(기본: 현재 시각). 같은 이름으로 재실행하면 이어서 함")

def start_collect(port, root):
    """결과 수집 서버를 같은 프로세스의 스레드로 띄움(한 번의 실행으로 FL 서버 + 수집기)."""
    import threading, subprocess, sys as _sys
    os.makedirs(root, exist_ok=True)
    proc = subprocess.Popen([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "collect_server.py"), "--port", str(port), "--dir", root],
                            stdout=open(os.path.join(root, "collect.log"), "a"), stderr=subprocess.STDOUT)
    return proc


def main():
    ap.add_argument("--methods", nargs="*", default=None); a = ap.parse_args()
    run = a.run or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    collect = start_collect(a.collect_port, "outputs/collected") if a.collect_port else None
    try:
        for cfg in a.config: run_experiment(cfg, a, run)
        if collect:   # 센터들의 최종 업로드는 서버 세션 종료 '후'에 오므로 수집기는 계속 대기 (outputs/STOP_SERVER.txt 로 종료)
            print(f"모든 실험 완료 — 결과 수집 서버(:{a.collect_port})는 계속 대기합니다. 종료: outputs/STOP_SERVER.txt 생성", flush=True)
            while not os.path.exists("outputs/STOP_SERVER.txt"): time.sleep(30)
    finally:
        if collect: collect.terminate()


def run_experiment(cfg_path, a, run):
    C = yaml.safe_load(open(cfg_path)); folds = a.folds if a.folds is not None else C.get("folds", [0, 1, 2, 3, 4]); methods = a.methods or C.get("methods", METHODS)
    out = os.path.join(C.get("output_dir", "outputs"), C["experiment"], run, "server"); os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "server.log"), "a")
    def log(s): print(s, flush=True); logf.write(f"{datetime.datetime.now():%m-%d %H:%M:%S} {s}\n"); logf.flush()

    model = build_model(); keys = list(model.state_dict().keys())
    if C.get("init_weights"):
        sd = torch.load(C["init_weights"], map_location="cpu", weights_only=False); sd = sd.get("model", sd); model.load_state_dict(sd); log(f"init: {C['init_weights']}")
    init_nd = get_nd(model); del model
    log(f"run={run}"); log(f"[{C['experiment']}] address {C['server_address']} | min_clients {C['min_clients']} | rounds {C['rounds']} x local {C['local_epochs']} | folds {folds} | methods {methods}")
    for fold in folds:
        for method in methods:
            done = os.path.join(out, f"fold{fold}", f"DONE_{method}")
            if os.path.exists(done): log(f"skip fold{fold} {method} (done)"); continue
            od = os.path.join(out, f"fold{fold}"); log(f"=== fold {fold} / {method} : {C['min_clients']}개 클라이언트 대기 ===")
            strat = CouinaudStrategy(method, keys, od, C["rounds"], C["local_epochs"], lr=C.get("lr", 0.01), fedprox_mu=C.get("fedprox_mu", 0.01),
                                     server_lr=C.get("fedadam_lr", 1e-3), fold=fold, fraction_fit=1.0, fraction_evaluate=1.0,
                                     min_fit_clients=C["min_clients"], min_evaluate_clients=C["min_clients"], min_available_clients=C["min_clients"],
                                     initial_parameters=ndarrays_to_parameters(init_nd))
            t0 = time.time()
            fl.server.start_server(server_address=C["server_address"], config=fl.server.ServerConfig(num_rounds=C["rounds"], round_timeout=float(C.get("round_timeout_sec", 10800))), strategy=strat)
            open(done, "w").write(f"{datetime.datetime.now()} {time.time()-t0:.0f}s\n"); log(f"fold {fold} {method} 완료 {time.time()-t0:.0f}s")
            time.sleep(C.get("gap_sec", 10))
    log("모든 세션 완료"); logf.close()


if __name__ == "__main__":
    main()