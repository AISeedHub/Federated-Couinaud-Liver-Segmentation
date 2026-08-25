#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  FedMorph Liver Segmentation — Start Server
# ══════════════════════════════════════════════════════════════

USE_CASE="liver_segmentation"

cd "$(dirname "$0")/.."

echo "============================================================"
echo " FedMorph Liver Segmentation Server"
echo "============================================================"
echo ""
echo "Starting server (waiting for clients to connect) ..."
echo ""

uv run python "src/use_cases/$USE_CASE/main_server.py" "$@"
