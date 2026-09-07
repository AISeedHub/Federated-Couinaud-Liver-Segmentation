#!/usr/bin/env bash
# 라운드 도중 클라이언트 강제 종료(정전 모사) → 서버가 멈추지 않고 진행 + 재시작한 클라이언트가 다음 라운드 합류하는지 검증
cd "$(dirname "$0")/.."; RUN=crash1; D=/data/campaign/couinaudfl_outputs/crash; rm -rf outputs/crash $D; mkdir -p $D; ln -sfn $D outputs/crash
P=.venv/bin/python; A=/data/campaign/_smoke_sites/A; B=/data/campaign/_smoke_sites/B
nohup $P scripts/server.py --config configs/crash.yaml --run $RUN --collect-port 0 > outputs/crash/server.out 2>&1 &
sleep 8
nohup $P scripts/client.py --config configs/crash.yaml --data $B --site B --workers 2 --run $RUN > outputs/crash/B.out 2>&1 &
nohup $P scripts/client.py --config configs/crash.yaml --data $A --site A --workers 2 --run $RUN > outputs/crash/A.out 2>&1 &
LA=outputs/crash/$RUN/client_A/client.log
# 라운드 1 평가가 끝나고 라운드 2 학습이 시작되면 A를 강제 종료
until grep -q '"round": 1' $LA 2>/dev/null; do sleep 5; done; sleep 20
PA=$(pgrep -f "scripts/client.py --config configs/crash.yaml --data $A"); echo "[$(date +%T)] kill -9 A (pid $PA) during round 2 fit" | tee -a outputs/crash/test.log; kill -9 $PA
sleep 40; echo "[$(date +%T)] restart A (run_center 재시작 모사)" | tee -a outputs/crash/test.log
nohup $P scripts/client.py --config configs/crash.yaml --data $A --site A --workers 2 --run $RUN > outputs/crash/A2.out 2>&1 &
until grep -q "모든 세션 완료" outputs/crash/$RUN/server/server.log 2>/dev/null; do sleep 10; done; sleep 20
echo "=== server ==="; grep -v -i warn outputs/crash/$RUN/server/server.log | tail -4
echo "=== A client.log ==="; grep -v -i warn $LA | grep -E "round|===|완료|대기|예외" | tail -8
echo "=== B client.log ==="; grep -E '"round"' outputs/crash/$RUN/client_B/client.log | tail -3 | cut -c1-120
echo "=== round_history ==="; $P -c "import json;h=json.load(open('outputs/crash/$RUN/server/fold0/round_history_FedAvg.json'));[print(r['round'], 'fit n=%s fail=%s'%(r.get('n_clients'),r.get('failures')) if 'n_clients' in r else 'eval clients=%s'%list(r['clients'])) for r in h]"
echo "=== DONE/test ==="; ls outputs/crash/$RUN/client_A/fold0/ outputs/crash/$RUN/client_A/fold0/FedAvg/ | tr '\n' ' '; echo; ls outputs/crash/$RUN/server/fold0/
