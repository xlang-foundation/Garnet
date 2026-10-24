@echo off
setlocal
set GARNET_RUNTIME_ROOT=D:\CantorAI\5090-study-20261023\build\bin
set CUDA_VISIBLE_DEVICES=0
set GARNET_NVFP4_DOWN_CANDIDATE=0
set GARNET_PROFILE_DECODE_LAYERS=
set GARNET_FP8_GDN_M32_CANONICAL=1
set GARNET_FP8_FULL_ATTN_M32_CANONICAL=1
set PATH=%GARNET_RUNTIME_ROOT%;D:\CantorAI\5090-study-20261023\tools\TensorRT\bin;D:\CantorAI\5090-study-20261023\tools\cuda13\nvidia\cu13\bin\x86_64;%PATH%
set "TMP=%~2"
set "TEMP=%TMP%"
cd /d D:\CantorAI\5090-study-20261023\Garnet
if not exist "%~1" mkdir "%~1"
set "GARNET_PROFILE_DECODE_LAYERS=%~1\preflight\layers.tsv"
%GARNET_RUNTIME_ROOT%\xlang3.exe D:/CantorAI/5090-study-20261023/Garnet/test2026/cuda/qwen38_5090/probe_trt_layer_profiler.py D:/CantorAI/5090-study-20261023 "%~1\preflight" --startup-profile
if errorlevel 1 exit /b %errorlevel%
set "GARNET_PROFILE_DECODE_LAYERS=%~1\layers.tsv"
%GARNET_RUNTIME_ROOT%\xlang3.exe D:/CantorAI/5090-study-20261023/Garnet/test2026/cuda/qwen38_5090/probe_b4_joined_layer_profile.py D:/CantorAI/5090-study-20261023 "%~1"
exit /b %errorlevel%
