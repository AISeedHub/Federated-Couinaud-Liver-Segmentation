@echo off
REM Windows 설치: CUDA 드라이버 버전에 맞는 torch 휠 선택. 잠금 파일 없음.
cd /d "%~dp0\.."
where uv >nul 2>nul || pip install -q uv
if not exist .venv uv venv --python 3.12 .venv
set CUDA=0.0
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "(nvidia-smi | Select-String 'CUDA Version: *([0-9]+\.[0-9]+)').Matches[0].Groups[1].Value"`) do set CUDA=%%v
for /f "tokens=1,2 delims=." %%a in ("%CUDA%") do (set MAJ=%%a& set MIN=%%b)
set IDX=cpu
if %MAJ% GEQ 13 set IDX=cu130
if %MAJ%==12 if %MIN% GEQ 8 set IDX=cu128
if %MAJ%==12 if %MIN% LSS 8 if %MIN% GEQ 6 set IDX=cu126
if %MAJ%==12 if %MIN% LSS 6 set IDX=cu121
echo CUDA driver %CUDA% -^> torch index %IDX%
REM 순서 중요: 의존성 먼저, torch는 마지막에 인덱스 강제(PyPI가 cu130으로 덮는 사고 방지)
uv pip install --python .venv -e .
uv pip install --python .venv --reinstall torch torchvision --index-url https://download.pytorch.org/whl/%IDX%
uv pip install --python .venv "numpy<2"
.venv\Scripts\python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
