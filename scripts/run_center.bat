@echo off
chcp 65001 >nul
REM 센터용 원커맨드 실행(Windows): 단일센터 5-fold → FL 클라이언트(fold×방법론 자동 순회).
REM 사용: run_center.bat <실험> <데이터폴더> <센터코드> [run이름|-] [spacing파일|-] [레이블맵json|-]
REM 실행마다 outputs\<실험>\<run>\ 에 별도 저장(기본 run = 시작 시각). 이어서 하려면 같은 run 이름을 4번째 인자로.
REM 종료: outputs\<실험>\client_<센터>\STOP.txt 생성(현 세션 후 정상 종료). 재실행 시 완료된 fold/method는 건너뜀.
setlocal
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0\.."
for /f %%f in ('powershell -NoProfile -Command "[math]::Floor((Get-PSDrive -Name (Get-Location).Drive.Name).Free/1GB)"') do set FREE_GB=%%f
if %FREE_GB% LSS 25 (echo [오류] 디스크 가용 %FREE_GB%GB ^< 25GB - 정리 후 재실행 ^(본 실험은 센터당 약 60~120GB 필요^) & exit /b 1)
if %FREE_GB% LSS 100 echo [경고] 디스크 가용 %FREE_GB%GB - 실험 도중 부족할 수 있음^(권장 120GB+^)
if "%~3"=="" (echo usage: run_center.bat exp4c D:\data\liver A & exit /b 1)
set "EXP=%~1"
set "DATA=%~2"
set "SITE=%~3"
set "RUN=%~4"
set "SPACING=%~5"
set "LABELMAP=%~6"
if "%SPACING%"=="-" set SPACING=
if "%LABELMAP%"=="-" set LABELMAP=
set SPACING_ARG=
if not "%SPACING%"=="" set SPACING_ARG=--spacing-file "%SPACING%"
set LABEL_ARG=
if not "%LABELMAP%"=="" set LABEL_ARG=--lesion-labels "%LABELMAP%"
if "%RUN%"=="" set RUN=%RUN_NAME%
if "%RUN%"=="" for /f "tokens=1-3 delims=/:. " %%a in ("%date% %time%") do set RUN=run_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set RUN=%RUN: =0%
set PYTHONIOENCODING=utf-8
REM 터미널 QuickEdit(클릭 시 정지) 해제 + 절전 방지
powershell -NoProfile -Command "$s=(Get-ItemProperty HKCU:\Console).QuickEdit; if($s -ne 0){Set-ItemProperty HKCU:\Console QuickEdit 0}" >nul 2>nul
powercfg /change standby-timeout-ac 0 >nul 2>nul
if not exist outputs\%EXP%\%RUN%\client_%SITE% mkdir outputs\%EXP%\%RUN%\client_%SITE%
echo %RUN%> outputs\%EXP%\LAST_RUN
set LOG=outputs\%EXP%\%RUN%\client_%SITE%\run_center.log
REM 실행 전 GPU 점검 (CPU 폴백 방지)
.venv\Scripts\python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>>%LOG%
if errorlevel 1 (echo [오류] GPU를 잡지 못했습니다. nvidia-smi 확인 후 scripts\install.bat 재실행 ^(uv sync/uv run 금지^) & echo GPU FAIL >> %LOG% & exit /b 1)
REM 서버 연결 선점검: FL 포트는 single 종료 후에야 처음 접속 - 미리 확인 (건너뛰려면 set SKIP_SERVER_CHECK=1)
if not "%SKIP_SERVER_CHECK%"=="1" (
  .venv\Scripts\python.exe scripts\check_server.py --exp %EXP%
  if errorlevel 1 (echo [오류] 서버 연결 점검 실패 - 서버 기동/방화벽/SERVER_IP 치환 확인 후 재실행 & echo SERVER CHECK FAIL >> %LOG% & exit /b 1)
)
REM 사전학습 가중치 자동 수급
for /f "tokens=2" %%w in ('findstr /b "init_weights:" configs\%EXP%.yaml') do set IW=%%w
set IW=%IW:"=%
for /f "tokens=2" %%u in ('findstr /b "upload_url:" configs\%EXP%.yaml') do set UURL=%%u
set UURL=%UURL:"=%
set UURL=%UURL:/upload=%
if not "%IW%"=="" if not exist "%IW%" (
  if not exist "%IW%\.." mkdir outputs\pretrain 2>nul
  echo [가중치] %IW% 없음 - %UURL%/weights 다운로드 >> %LOG%
  curl -fsSL -o "%IW%" "%UURL%/weights" || curl -fsSL -o "%IW%" "https://github.com/AISeedHub/Federated-Couinaud-Liver-Segmentation/releases/download/weights-v2.0/best.pth" || (echo [오류] 사전학습 가중치 다운로드 실패 - 수동 배치 필요 & exit /b 1)
)
echo [%date% %time%] start %EXP% %SITE% >> %LOG%
:loop
if exist outputs\%EXP%\%RUN%\client_%SITE%\STOP.txt (echo STOP.txt — exit >> %LOG% & goto end)
.venv\Scripts\python scripts\patient_catalog.py --data "%DATA%" --exp %EXP% --site %SITE% --run %RUN% %SPACING_ARG% %LABEL_ARG% >> %LOG% 2>&1
set RS=1
for /f "tokens=2" %%r in ('findstr /b "run_single:" configs\%EXP%.yaml') do set RS=%%r
if "%RS%"=="0" goto fl
.venv\Scripts\python scripts\single.py --config configs\%EXP%.yaml --data "%DATA%" --site %SITE% --run %RUN% %SPACING_ARG% >> %LOG% 2>&1
.venv\Scripts\python -c "import sys;sys.exit(0 if '단일센터 완료' in open(r'outputs/%EXP%/%RUN%/client_%SITE%/single.log',encoding='utf-8',errors='ignore').read() else 1)" || (echo [%date% %time%] single.py 미완료 - 60s 후 재시도 >> %LOG% & timeout /t 60 /nobreak >nul & goto loop)
:fl
REM 진짜 로컬(단일센터) 모델·지표를 FL 전에 먼저 서버로 전송
.venv\Scripts\python scripts\export_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1 && .venv\Scripts\python scripts\upload_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
.venv\Scripts\python scripts\client.py --config configs\%EXP%.yaml --data "%DATA%" --site %SITE% --run %RUN% %SPACING_ARG% >> %LOG% 2>&1
if exist outputs\%EXP%\%RUN%\client_%SITE%\STOP.txt goto end
.venv\Scripts\python -c "import sys;sys.exit(0 if '모든 세션 완료' in open(r'%LOG%',encoding='utf-8',errors='ignore').read() else 1)" && goto end
echo [%date% %time%] process exited unexpectedly — restart in 60s >> %LOG%
timeout /t 60 /nobreak >nul
goto loop
:end
REM 완료 후 자동: 익명 내보내기 → 서버 업로드(실패 시 zip만 남김)
.venv\Scripts\python scripts\export_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
.venv\Scripts\python scripts\upload_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
echo [%date% %time%] done >> %LOG%
endlocal
