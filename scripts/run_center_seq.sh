#!/usr/bin/env bash
# 여러 실험을 순서대로: run_center_seq.sh <데이터폴더> <센터코드> exp4c exp5c ...
#   nohup bash scripts/run_center_seq.sh /data/merged A exp4c exp5c > /dev/null 2>&1 &
cd "$(dirname "$0")/.."; DATA=${1:?data}; SITE=${2:?site}; shift 2
for EXP in "$@"; do bash scripts/run_center.sh "$EXP" "$DATA" "$SITE"; done
