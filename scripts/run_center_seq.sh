#!/usr/bin/env bash
# 여러 실험 순차 + 부속 파일 명시: run_center_seq.sh <데이터폴더> <센터코드> [spacing파일] [레이블맵.json] exp4c exp5c ...
#   파일 인자는 순서 무관 자동 판별(.json=레이블맵, 그 외=spacing). 생략하면 데이터 폴더 규약/내장 매핑 사용.
#   nohup bash scripts/run_center_seq.sh /data/merged A exp4c exp5c > /dev/null 2>&1 &
cd "$(dirname "$0")/.."; DATA=${1:?data}; SITE=${2:?site}; shift 2
SPACING="-"; LABELMAP="-"; EXPS=""
for a in "$@"; do
  if [ -f "configs/$a.yaml" ]; then EXPS="$EXPS $a"
  elif [ -f "$a" ]; then
    case "$a" in *.json) LABELMAP="$a";; *) SPACING="$a";; esac
  else echo "알 수 없는 인자(실험 yaml도 파일도 아님): $a"; exit 1; fi
done
for EXP in $EXPS; do bash scripts/run_center.sh "$EXP" "$DATA" "$SITE" "${RUN_NAME:-}" "$SPACING" "$LABELMAP"; done
