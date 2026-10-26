@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
set PYTHONDONTWRITEBYTECODE=1
set "PATH=%STUDY_ROOT%\build\bin;%PATH%"
%STUDY_ROOT%\build\bin\xlang3.exe %STUDY_ROOT%/Garnet/test2026/cuda/qwen38_5090/probe_fp8_z128_ctypes_bound.py %STUDY_ROOT% "%~1"
exit /b %errorlevel%
