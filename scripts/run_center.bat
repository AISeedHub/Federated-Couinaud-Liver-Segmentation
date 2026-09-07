@echo off
REM 센터용 원커맨드 실행(Windows): 단일센터 5-fold → FL 클라이언트(fold×방법론 자동 순회).
REM 사용: run_center.bat <실험(exp4c|exp5c)> <데이터폴더> <센터코드(A|B|C|D|E)> [run이름]
REM 실행마다 outputs\<실험>\<run>\ 에 별도 저장(기본 run = 시작 시각). 이어서 하려면 같은 run 이름을 4번째 인자로.
REM 종료: outputs\<실험>\client_<센터>\STOP.txt 생성(현 세션 후 정상 종료). 재실행 시 완료된 fold/method는 건너뜀.
setlocal
cd /d "%~dp0\.."
if "%~3"=="" (echo usage: run_center.bat exp4c D:\data\liver A & exit /b 1)
set EXP=%~1& set DATA=%~2& set SITE=%~3& set RUN=%~4
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
echo [%date% %time%] start %EXP% %SITE% >> %LOG%
:loop
if exist outputs\%EXP%\%RUN%\client_%SITE%\STOP.txt (echo STOP.txt — exit >> %LOG% & goto end)
.venv\Scripts\python scripts\patient_catalog.py --data "%DATA%" --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
set RS=1
for /f "tokens=2" %%r in ('findstr /b "run_single:" configs\%EXP%.yaml') do set RS=%%r
if "%RS%"=="0" goto fl
.venv\Scripts\python scripts\single.py --config configs\%EXP%.yaml --data "%DATA%" --site %SITE% --run %RUN% >> %LOG% 2>&1
findstr /c:"단일센터 완료" outputs\%EXP%\%RUN%\client_%SITE%\single.log >nul || (echo [%date% %time%] single.py 미완료 - 60s 후 재시도 >> %LOG% & timeout /t 60 /nobreak >nul & goto loop)
:fl
REM 진짜 로컬(단일센터) 모델·지표를 FL 전에 먼저 서버로 전송
.venv\Scripts\python scripts\export_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1 && .venv\Scripts\python scripts\upload_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
.venv\Scripts\python scripts\client.py --config configs\%EXP%.yaml --data "%DATA%" --site %SITE% --run %RUN% >> %LOG% 2>&1
if exist outputs\%EXP%\%RUN%\client_%SITE%\STOP.txt goto end
findstr /c:"모든 세션 완료" %LOG% >nul && goto end
echo [%date% %time%] process exited unexpectedly — restart in 60s >> %LOG%
timeout /t 60 /nobreak >nul
goto loop
:end
REM 완료 후 자동: 익명 내보내기 → 서버 업로드(실패 시 zip만 남김)
.venv\Scripts\python scripts\export_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
.venv\Scripts\python scripts\upload_results.py --exp %EXP% --site %SITE% --run %RUN% >> %LOG% 2>&1
echo [%date% %time%] done >> %LOG%
endlocal
