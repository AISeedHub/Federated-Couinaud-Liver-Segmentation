@echo off
chcp 65001 >nul
REM 여러 실험 순차 + 부속 파일 명시: run_center_seq.bat <데이터폴더> <센터코드> [spacing파일] [레이블맵.json] exp4c exp5c ...
REM   파일 인자는 순서 무관 자동 판별(.json=레이블맵, 그 외 실존 파일=spacing). 생략 시 데이터 폴더 규약/내장 매핑 사용.
cd /d "%~dp0\.."
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "DATA=%~1"
set "SITE=%~2"
shift & shift
set SPACING=-& set LABELMAP=-& set EXPS=
:scan
if "%~1"=="" goto run
if exist "configs\%~1.yaml" (set EXPS=%EXPS% %~1& goto nextarg)
if exist "%~1" (
  if /i "%~x1"==".json" (set "LABELMAP=%~1") else (set "SPACING=%~1")
  goto nextarg
)
echo 알 수 없는 인자: %~1 & exit /b 1
:nextarg
shift
goto scan
:run
if "%EXPS%"=="" (echo 실험 이름이 없습니다 & exit /b 1)
for %%E in (%EXPS%) do call scripts\run_center.bat %%E "%DATA%" %SITE% "%RUN_NAME%" "%SPACING%" "%LABELMAP%"
