#!/usr/bin/env bash
# 최종 통합 테스트(MR 포함): seq1(CT 2센터) → seq2mr(CT 2 + MR E, min_clients 3)
cd "$(dirname "$0")/.."; RUN=${1:-seqmr}; export RUN_NAME=$RUN
for s in seq1 seq2mr; do D=/data/campaign/couinaudfl_outputs/$s; rm -rf $D/$RUN; mkdir -p $D; [ -L outputs/$s ] || rm -rf outputs/$s; ln -sfn $D outputs/$s; done
rm -rf outputs/collected/seq1/$RUN outputs/collected/seq2mr/$RUN
P=.venv/bin/python; A=/data/campaign/_rehearsal_sites/A; B=/data/campaign/_rehearsal_sites/B; E=/data/campaign/e_center_mr
nohup $P scripts/server.py --config configs/seq1.yaml configs/seq2mr.yaml --run $RUN > outputs/seq1/server_$RUN.out 2>&1 &
nohup bash scripts/run_center.sh seq2mr $E E $RUN > /dev/null 2>&1 &     # E: MR, 처음부터 대기
sleep 10
nohup bash scripts/run_center_seq.sh $A A seq1 seq2mr > /dev/null 2>&1 &
nohup bash scripts/run_center_seq.sh $B B seq1 seq2mr > /dev/null 2>&1 &
until grep -q "모든 세션 완료" outputs/seq2mr/$RUN/server/server.log 2>/dev/null; do sleep 30; done; sleep 90
echo "=== 서버 ==="; grep -E "완료|===" outputs/seq1/$RUN/server/server.log | grep -v -i warn | tail -2; grep -E "완료|===" outputs/seq2mr/$RUN/server/server.log | grep -v -i warn | tail -3
echo "=== E(MR) 클라이언트 ==="; grep -E "patients|===|round|대기 횟수|완료" outputs/seq2mr/$RUN/client_E/client.log 2>/dev/null | grep -v -i warn | head -6 | cut -c1-120
echo "연결 대기 횟수: $(grep -c '연결 대기' outputs/seq2mr/$RUN/client_E/client.log 2>/dev/null)"
echo "=== 검증 seq2mr ==="; $P scripts/verify_run.py --config configs/seq2mr.yaml --sites A B E --data-roots /data/campaign/_rehearsal_sites/A /data/campaign/_rehearsal_sites/B /data/campaign/e_center_mr --run $RUN 2>&1 | grep -v -i warn | tail -6
echo "=== E 산출물·병변 처리 ==="; ls outputs/seq2mr/$RUN/client_E/fold0/FedAvg/ 2>/dev/null | tr '\n' ' '; echo
echo "E lesion_overlap 존재?: $(find outputs/seq2mr/$RUN/client_E -name lesion_overlap.csv | wc -l) (0이어야 정상 — MR 병변 채널 없음)"
echo "A lesion_overlap: $(find outputs/seq1/$RUN/client_A -name lesion_overlap.csv | wc -l)개"
echo "E catalog spacing_source: $(head -2 outputs/seq2mr/$RUN/client_E/patient_catalog.csv | tail -1 | cut -d, -f6)"
echo "=== 수집 ==="; for S in A B E; do D2=$(ls -d outputs/collected/seq2mr/$RUN/$S/*/ 2>/dev/null | tail -1); [ -n "$D2" ] && echo "$S: $(find $D2 -type f | wc -l) files" || echo "$S: 수집물 없음"; done
