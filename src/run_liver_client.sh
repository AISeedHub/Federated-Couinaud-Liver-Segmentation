#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  FedMorph Liver Segmentation — Start Client
# ══════════════════════════════════════════════════════════════
#
#  Usage:
#    ./run_liver_client.sh SERVER_ADDRESS [DATA_DIR]
#
#  Examples:
#    ./run_liver_client.sh 192.168.1.100:443
#    ./run_liver_client.sh 192.168.1.100:443 /data/hospital_a/ct
# ══════════════════════════════════════════════════════════════

USE_CASE="liver_segmentation"

if [ $# -eq 0 ]; then
    echo ""
    echo "  Usage: run_liver_client.sh SERVER_ADDRESS [DATA_DIR]"
    echo ""
    echo "  SERVER_ADDRESS : e.g. 192.168.1.100:443"
    echo "  DATA_DIR       : e.g. /data/liver_ct  (optional, overrides config)"
    echo ""
    exit 1
fi

SERVER_ADDR=$1
DATA_DIR=$2

cd "$(dirname "$0")/.."

echo "============================================================"
echo " FedMorph Liver Segmentation Client"
echo "============================================================"

if [ -z "$DATA_DIR" ]; then
    echo "Starting client connecting to $SERVER_ADDR ..."
    uv run python "src/use_cases/$USE_CASE/main_client.py" --server-address "$SERVER_ADDR"
else
    echo "Starting client connecting to $SERVER_ADDR ..."
    echo "Data dir: $DATA_DIR"
    uv run python "src/use_cases/$USE_CASE/main_client.py" --server-address "$SERVER_ADDR" --data-dir "$DATA_DIR"
fi
