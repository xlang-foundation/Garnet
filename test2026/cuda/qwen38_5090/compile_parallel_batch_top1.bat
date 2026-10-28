@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
set "TMP=%~3"
set "TEMP=%TMP%"
cd /d "%~3"
where cl > "%~3\host-compiler.txt"
if errorlevel 1 exit /b %errorlevel%
echo VCToolsVersion=%VCToolsVersion%
echo WindowsSDKVersion=%WindowsSDKVersion%
"%~2" --version
if errorlevel 1 exit /b %errorlevel%
"%~2" -std=c++17 -arch=sm_120 -O3 -ccbin "%~4" -Xcompiler=/MT,/utf-8 "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_parallel_batch_top1.cu" "D:\CantorAI\5090-study-20261023\Garnet\src\cuda\logits_sampler_kernel.cu" -o "%~3\probe.exe"
exit /b %errorlevel%
