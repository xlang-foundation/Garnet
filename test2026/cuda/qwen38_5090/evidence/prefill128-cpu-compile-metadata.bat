@echo off
call "C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat" -vcvars_ver=14.29
if errorlevel 1 exit /b %errorlevel%
set "TMP=%~dp0"
set "TEMP=%~dp0"
cl /nologo /EHsc /std:c++17 /MT /utf-8 /I"D:/CantorAI/work/qwen38-5090/third_party" /I"D:/CantorAI/work/qwen38-5090/src/backends/tensorrt" /I"D:/CantorAI/work/qwen38-5090/src/cuda" /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/include" "D:/CantorAI/work/qwen38-5090/test2026/cuda/qwen38_5090/probe_bf16_prefill128_metadata.cpp" /Fo"%~dp0metadata.obj" /Fd"%~dp0metadata.pdb" /Fe"%~dp0metadata.exe"
if errorlevel 1 exit /b %errorlevel%
dumpbin /imports "%~dp0metadata.exe" > "%~dp0metadata-imports.txt"
exit /b %errorlevel%
