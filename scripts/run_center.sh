#!/usr/bin/env bash
# 센터용 원커맨드 실행(Linux, 예: DGX Spark): 단일센터 5-fold → FL 클라이언트. nohup으로 띄우면 세션이 끊겨도 계속 돈다.
#   nohup bash scripts/run_center.sh exp5c /home/user/merged E [run이름] [spacing파일|-] [레이블맵json|-] > /dev/null 2>&1 &
# 실행마다 outputs/<exp>/<run>/ 에 별도 저장(기본 run = 시작 시각). 이어서 하려면 같은 run 이름을 4번째 인자로.
# 종료: outputs/<exp>/client_<site>/STOP.txt 생성. 재실행 시 완료된 fold/method 건너뜀.
set -u; cd "$(dirname "$0")/.."
FREE_GB=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
if [ "$FREE_GB" -lt 25 ]; then echo "[오류] 디스크 가용 ${FREE_GB}GB < 25GB — 정리 후 재실행 (본 실험은 센터당 약 60~120GB 필요)"; exit 1; fi
if [ "$FREE_GB" -lt 100 ]; then echo "[경고] 디스크 가용 ${FREE_GB}GB — 실험 도중 부족할 수 있음(권장 120GB+)"; fi
EXP=${1:?exp}; DATA=${2:?data dir}; SITE=${3:?site}; RUN=${4:-${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}}
SPACING=${5:-}; LABELMAP=${6:-}; [ "$SPACING" = "-" ] && SPACING=""; [ "$LABELMAP" = "-" ] && LABELMAP=""
SPACING_ARG=""; [ -n "$SPACING" ] && SPACING_ARG="--spacing-file $SPACING"
LABEL_ARG="";  [ -n "$LABELMAP" ] && LABEL_ARG="--lesion-labels $LABELMAP"
OUT=outputs/$EXP/$RUN/client_$SITE; mkdir -p "$OUT"; LOG=$OUT/run_center.log; echo "$RUN" > outputs/$EXP/LAST_RUN
# 실행 전 GPU 점검: CPU 폴백 상태로 며칠 도는 사고 방지 (v1 사례: cu130 휠 + 드라이버 12.6)
if ! .venv/bin/python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>>"$LOG"; then
  echo "[오류] GPU를 잡지 못했습니다. nvidia-smi의 CUDA 버전 확인 후 scripts/install.sh 재실행 (uv sync/uv run 사용 금지 — torch가 제거됨)" | tee -a "$LOG"; exit 1
fi
# 사전학습 가중치 자동 수급 (없으면 서버 9598에서 다운로드)
IW=$(grep -E "^init_weights:" configs/$EXP.yaml | awk '{print $2}' | tr -d '"')
UURL=$(grep -E "^upload_url:" configs/$EXP.yaml | awk '{print $2}' | tr -d '"' | sed 's|/upload$||')
if [ -n "$IW" ] && [ ! -s "$IW" ] && [ -n "$UURL" ]; then
  mkdir -p "$(dirname "$IW")"; echo "[가중치] $IW 없음 → $UURL/weights 다운로드" >> "$LOG"
  curl -fsSL -o "$IW" "$UURL/weights" >> "$LOG" 2>&1 \
    || curl -fsSL -o "$IW" "https://github.com/AISeedHub/Federated-Couinaud-Liver-Segmentation/releases/download/weights-v2.0/best.pth" >> "$LOG" 2>&1 \
    || { echo "[오류] 사전학습 가중치 다운로드 실패(서버·GitHub 모두) — 수동으로 $IW 에 배치 필요" | tee -a "$LOG"; exit 1; }
fi
echo "[$(date)] start $EXP $SITE run=$RUN" >> "$LOG"
while true; do
  [ -f "$OUT/STOP.txt" ] && { echo "STOP.txt — exit" >> "$LOG"; break; }
  .venv/bin/python scripts/patient_catalog.py --data "$DATA" --exp $EXP --site "$SITE" --run "$RUN" $SPACING_ARG $LABEL_ARG >> "$LOG" 2>&1
  RS=$(grep -E "^run_single:" configs/$EXP.yaml | awk '{print $2}'); RS=${RS:-1}
  if [ "$RS" != "0" ]; then
    .venv/bin/python scripts/single.py --config configs/$EXP.yaml --data "$DATA" --site "$SITE" --run "$RUN" $SPACING_ARG >> "$LOG" 2>&1
    if ! grep -q "단일센터 완료" "$OUT/single.log" 2>/dev/null; then echo "[$(date)] single.py 미완료 — 60s 후 재시도" >> "$LOG"; sleep 60; continue; fi
  fi
  # 진짜 로컬(단일센터) 모델·지표를 FL 전에 먼저 서버로 전송
  .venv/bin/python scripts/export_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1 && .venv/bin/python scripts/upload_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1
  .venv/bin/python scripts/client.py --config configs/$EXP.yaml --data "$DATA" --site "$SITE" --run "$RUN" $SPACING_ARG >> "$LOG" 2>&1
  [ -f "$OUT/STOP.txt" ] && break
  grep -q "모든 세션 완료" "$LOG" && break
  echo "[$(date)] process exited unexpectedly — restart in 60s" >> "$LOG"; sleep 60
done
# 완료 후 자동: 익명 내보내기 → 서버 업로드(실패 시 zip만 남김)
.venv/bin/python scripts/export_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1
.venv/bin/python scripts/upload_results.py --exp $EXP --site "$SITE" --run "$RUN" >> "$LOG" 2>&1
echo "[$(date)] done" >> "$LOG"
