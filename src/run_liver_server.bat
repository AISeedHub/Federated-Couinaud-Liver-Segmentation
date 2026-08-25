@echo off
rem ══════════════════════════════════════════════════════════
rem  FedMorph Liver Segmentation — Start Server
rem ══════════════════════════════════════════════════════════

set USE_CASE=liver_segmentation

rem Move to project root
cd /d "%~dp0\.."

echo ============================================================
echo  FedMorph Liver Segmentation Server
echo ============================================================
echo.

rem Optional: override data paths via env vars
rem set FEDMORPH_DATA_DIR=D:\data\combined
rem set FEDMORPH_CLIENT_DATA_DIR=D:\data\fl_clients

echo Starting server (waiting for 3 clients to connect) ...
echo.
uv run python src/use_cases/%USE_CASE%/main_server.py %*

pause
