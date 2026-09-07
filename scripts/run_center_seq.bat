@echo off
REM 여러 실험을 순서대로: run_center_seq.bat <데이터폴더> <센터코드> exp4c exp5c ...
cd /d "%~dp0\.."
set DATA=%~1& set SITE=%~2
shift & shift
:next
if "%~1"=="" goto done
call scripts\run_center.bat %~1 "%DATA%" %SITE%
shift
goto next
:done
