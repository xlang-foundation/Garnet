@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
set CUDA_VISIBLE_DEVICES=0
set "PATH=%STUDY_ROOT%\tools\TensorRT\bin;%STUDY_ROOT%\tools\cuda13\nvidia\cu13\bin\x86_64;%PATH%"
"%STUDY_ROOT%\build\bin\garnet_canonical_kv_overlay_plugin_test.exe" "%STUDY_ROOT%\results\attention-kv-delta-strided-attempt1" "%~1"
exit /b %errorlevel%
