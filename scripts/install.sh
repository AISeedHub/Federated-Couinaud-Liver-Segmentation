#!/usr/bin/env bash
# Linux 설치: CUDA 드라이버 버전을 읽어 맞는 torch 휠 인덱스를 고른다. 잠금 파일 없음(머신별 자율 해석).
set -e
cd "$(dirname "$0")/.."
command -v uv >/dev/null || pip install -q uv
[ -d .venv ] || uv venv --python 3.12 .venv 2>/dev/null || uv venv .venv
CUDA=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$" || echo "0")
MAJ=${CUDA%%.*}; MIN=${CUDA#*.}; MIN=${MIN%%.*}
if [ "$MAJ" -ge 13 ]; then IDX=cu130; elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; then IDX=cu128; elif [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 6 ]; then IDX=cu126; elif [ "$MAJ" -eq 12 ]; then IDX=cu121; else IDX=cpu; fi
ARCH=$(uname -m); [ "$ARCH" = "aarch64" ] && [ "$IDX" != "cu130" ] && IDX=cu130   # DGX Spark 등 ARM은 cu130 휠만 제공
echo "CUDA driver $CUDA → torch index $IDX ($ARCH)"
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/$IDX
uv pip install --python .venv -e .
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
