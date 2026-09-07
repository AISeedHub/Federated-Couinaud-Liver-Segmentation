#!/usr/bin/env bash
# 정식 모의시험: 서버 + 모의 센터 2개를 실제 배포 경로(run_center.sh)로 실행 → verify_run.py
cd "$(dirname "$0")/.."; RUN=${1:-run_$(date +%Y%m%d_%H%M%S)}; mkdir -p outputs/rehearsal/$RUN
sed -i 's/^client_server_address:.*/client_server_address: "127.0.0.1:9597"/' configs/rehearsal.yaml
nohup .venv/bin/python scripts/server.py --config configs/rehearsal.yaml --run $RUN > outputs/rehearsal/$RUN/server.out 2>&1 &
sleep 8
nohup bash scripts/run_center.sh rehearsal /data/campaign/_rehearsal_sites/A A $RUN > /dev/null 2>&1 &
nohup bash scripts/run_center.sh rehearsal /data/campaign/_rehearsal_sites/B B $RUN > /dev/null 2>&1 &
until grep -q "모든 세션 완료" outputs/rehearsal/$RUN/server/server.log 2>/dev/null; do sleep 30; done
sleep 30
.venv/bin/python scripts/verify_run.py --config configs/rehearsal.yaml --sites A B --data-roots /data/campaign/_rehearsal_sites/A /data/campaign/_rehearsal_sites/B --run $RUN | tee outputs/rehearsal/$RUN/verify.txt
