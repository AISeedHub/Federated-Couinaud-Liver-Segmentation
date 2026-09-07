#!/usr/bin/env bash
# 4센터→5센터 모사: A,B는 seq1→seq2 연속, C는 처음부터 seq2(min_clients 3)에서 대기 후 합류
cd "$(dirname "$0")/.."; RUN=${1:-seqtest3}; export RUN_NAME=$RUN
for s in seq1 seq2; do D=/data/campaign/couinaudfl_outputs/$s; rm -rf $D/$RUN; mkdir -p $D; ln -sfn $D outputs/$s; done; rm -rf outputs/collected/seq1/$RUN outputs/collected/seq2/$RUN
P=.venv/bin/python; A=/data/campaign/_rehearsal_sites/A; B=/data/campaign/_rehearsal_sites/B; C=/data/campaign/_smoke_sites/A
# 서버 수집기는 9598 단독 수집기와 충돌하므로 --collect-port 0 (수집 자체는 seqtest2에서 검증됨)
nohup $P scripts/server.py --config configs/seq1.yaml configs/seq2.yaml --run $RUN --collect-port 0 > outputs/seq1/server_$RUN.out 2>&1 &
# C를 가장 먼저 켜서 seq2 포트(9591)가 열릴 때까지 대기하게 함
nohup bash scripts/run_center.sh seq2 $C C $RUN > /dev/null 2>&1 &
sleep 10
nohup bash scripts/run_center_seq.sh $A A seq1 seq2 > /dev/null 2>&1 &
nohup bash scripts/run_center_seq.sh $B B seq1 seq2 > /dev/null 2>&1 &
until grep -q "모든 세션 완료" outputs/seq2/$RUN/server/server.log 2>/dev/null; do sleep 30; done; sleep 60
echo "=== seq1 server ==="; grep -v -i warn outputs/seq1/$RUN/server/server.log | grep -E "완료|===" | tail -3
echo "=== seq2 server ==="; grep -v -i warn outputs/seq2/$RUN/server/server.log | grep -E "완료|===" | tail -3
echo "=== C client (대기 → 합류) ==="; L=outputs/seq2/$RUN/client_C/client.log; echo "연결 대기 횟수: $(grep -c '연결 대기' $L)"; grep -E "===|\"round\"|완료" $L | grep -v Warn | head -6 | cut -c1-110
echo "=== seq2 round_history 참여 클라이언트 ==="; $P -c "import json;h=json.load(open('outputs/seq2/$RUN/server/fold0/round_history_FedAvg.json'));[print(r['round'],'fit n=%s fail=%s'%(r.get('n_clients'),r.get('failures')) if 'n_clients' in r else 'eval %s'%sorted(r['clients'])) for r in h]"
echo "=== DONE ==="; for S in A B C; do echo "$S seq2: $(ls outputs/seq2/$RUN/client_$S/fold0/ 2>/dev/null | grep -c DONE_) DONE"; done; for S in A B; do echo "$S seq1: $(ls outputs/seq1/$RUN/client_$S/fold0/ 2>/dev/null | grep -c DONE_) DONE"; done
