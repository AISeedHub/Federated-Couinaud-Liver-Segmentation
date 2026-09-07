#!/usr/bin/env bash
# 순차 실행 통합 테스트: server.py(seq1→seq2, 수집기 내장) + 모의센터 2개(run_center_seq) → 검증 + 수집물 대조
cd "$(dirname "$0")/.."; RUN=${1:-seqtest}; export RUN_NAME=$RUN   # 두 모의센터가 같은 run 이름을 쓰도록
for p in $(pgrep -f "collect_server.p[y]"); do kill $p; done; sleep 1      # 단독 수집기 종료 → server.py 내장 수집기가 9598 사용
rm -rf outputs/seq1 outputs/seq2 outputs/collected/seq1 outputs/collected/seq2; mkdir -p outputs/seq1
nohup .venv/bin/python scripts/server.py --config configs/seq1.yaml configs/seq2.yaml --run $RUN > outputs/seq1/server_$RUN.out 2>&1 &
sleep 8
nohup bash scripts/run_center_seq.sh /data/campaign/_rehearsal_sites/A A seq1 seq2 > /dev/null 2>&1 &
nohup bash scripts/run_center_seq.sh /data/campaign/_rehearsal_sites/B B seq1 seq2 > /dev/null 2>&1 &
# run_center_seq는 RUN을 자동 생성하므로, 각 실험의 LAST_RUN으로 찾는다
until [ -f outputs/seq2/LAST_RUN ] && grep -q "\] done" outputs/seq2/$(cat outputs/seq2/LAST_RUN)/client_A/run_center.log 2>/dev/null && grep -q "\] done" outputs/seq2/$(cat outputs/seq2/LAST_RUN)/client_B/run_center.log 2>/dev/null; do sleep 30; done
echo "=== server log ==="; grep -v -i warn outputs/seq1/$RUN/server/server.log | tail -3; grep -v -i warn outputs/seq2/$RUN/server/server.log | tail -3
for E in seq1 seq2; do R=$(cat outputs/$E/LAST_RUN); echo "=== verify $E ($R) ==="
  .venv/bin/python scripts/verify_run.py --config configs/$E.yaml --sites A B --data-roots /data/campaign/_rehearsal_sites/A /data/campaign/_rehearsal_sites/B --run $R 2>&1 | grep -v -i warn | tail -8
  echo "--- collected $E:"; find outputs/collected/$E -maxdepth 4 -type d | sed 's|outputs/collected/||' | head; 
  for S in A B; do D=$(ls -d outputs/collected/$E/$R/$S/*/ 2>/dev/null | tail -1); [ -n "$D" ] && echo "  $S: $(find $D -type f | wc -l) files | logs $(find $D -name '*.log' | wc -l) | pth $(find $D -name '*.pth' | wc -l) | npz $(find $D -name '*.npz' | wc -l) | csv $(find $D -name '*.csv' | wc -l)" || echo "  $S: 수집물 없음"; done
done
