@echo off
setlocal
set GARNET_RUNTIME_ROOT=D:\CantorAI\5090-study-20261023\build\bin
set PATH=%GARNET_RUNTIME_ROOT%;%PATH%
cd /d D:\CantorAI\5090-study-20261023\Garnet
%GARNET_RUNTIME_ROOT%\xlang3.exe D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_native_stdio_cpu.py "%~1"
exit /b %errorlevel%
