@echo off
rem ══════════════════════════════════════════════════════════
rem  FedMorph Liver Segmentation — Start Client
rem ══════════════════════════════════════════════════════════
rem
rem  Usage:
rem    run_liver_client.bat SERVER_ADDRESS [DATA_DIR]
rem
rem  Examples:
rem    run_liver_client.bat 192.168.1.100:443
rem    run_liver_client.bat 192.168.1.100:443 D:\data\liver_ct
rem ══════════════════════════════════════════════════════════

set USE_CASE=liver_segmentation

if "%1"=="" (
    echo.
    echo  Usage: run_liver_client.bat SERVER_ADDRESS [DATA_DIR]
    echo.
    echo  SERVER_ADDRESS : e.g. 192.168.1.100:443
    echo  DATA_DIR       : e.g. D:\data\liver_ct  (optional, overrides config)
    echo.
    pause
    exit /b 1
)

set SERVER_ADDR=%1
set DATA_DIR=%2

rem Move to project root
cd /d "%~dp0\.."

echo ============================================================
echo  FedMorph Liver Segmentation Client
echo ============================================================
echo.

if "%DATA_DIR%"=="" (
    echo Starting client connecting to %SERVER_ADDR% ...
    uv run python src/use_cases/%USE_CASE%/main_client.py --server-address %SERVER_ADDR%
) else (
    echo Starting client connecting to %SERVER_ADDR% ...
    echo Data dir: %DATA_DIR%
    uv run python src/use_cases/%USE_CASE%/main_client.py --server-address %SERVER_ADDR% --data-dir %DATA_DIR%
)

pause
