@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29 >"%~dp0toolchain.log" 2>&1
if errorlevel 1 exit /b %errorlevel%
set "TMP=%~dp0"
set "TEMP=%TMP%"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" --std c++17 -arch=sm_120 -O3 -Xcompiler /MT -c "D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090\probe_parallel_batch_top1.cu" -o "%~dp0probe.obj" >"%~dp0compile.log" 2>&1
exit /b %errorlevel%
