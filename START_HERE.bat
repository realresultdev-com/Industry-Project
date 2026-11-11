@echo off
title Autonomous Industrial Defect Inspection System - Launcher
color 0A
cls
echo =========================================================================
echo    AUTONOMOUS INDUSTRIAL DEFECT DETECTION (SELF-SUPERVISED AI)
echo =========================================================================
echo.
echo  Checking system and launching setup...
echo.

:: Run the automated PowerShell installer & launcher with ExecutionPolicy Bypass
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_and_run.ps1"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =========================================================================
    echo  [!] An error occurred during launch.
    echo  If Python was not installed, please download it from https://python.org
    echo =========================================================================
    pause
)
