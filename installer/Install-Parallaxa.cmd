@echo off
rem Double-click to install or update Parallaxa for this user (no admin rights needed).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
pause
