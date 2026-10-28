@echo off
setlocal
cd /d D:\CantorAI\5090-study-20261023\Garnet
D:\CantorAI\5090-study-20261023\build\bin\xlang3.exe D:/CantorAI/5090-study-20261023/Garnet/test2026/cuda/qwen38_5090/probe_q128_model_metadata.py "%~1"
exit /b %errorlevel%
