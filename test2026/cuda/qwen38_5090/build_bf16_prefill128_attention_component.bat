@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
set "TASK_ROOT=D:\CantorAI\5090-study-20261023"
set "TASK_METHODS=%TASK_ROOT%\Garnet\test2026\cuda\qwen38_5090"
set "TASK_BUILD=%TASK_ROOT%\tools\bf16-kv-native"
set "TASK_NVCC=%TASK_ROOT%\tools\cuda13\nvidia\cu13\bin\nvcc.exe"
if exist "%TASK_BUILD%\prefill128-attention-probe.exe" exit /b 2
if exist "%TASK_BUILD%\prefill128-attention-core.obj" exit /b 2
if exist "%TASK_BUILD%\prefill128-attention-probe.obj" exit /b 2
set "TMP=%TASK_BUILD%"
set "TEMP=%TASK_BUILD%"
echo VCToolsVersion=%VCToolsVersion%
echo WindowsSDKVersion=%WindowsSDKVersion%
"%TASK_NVCC%" --version
if errorlevel 1 exit /b %errorlevel%
"%TASK_NVCC%" -std=c++17 -arch=sm_120 -O3 -Xcompiler=/MT,/utf-8 -Xptxas=-v -I"%TASK_ROOT%\Garnet\src\cuda" -c "%TASK_METHODS%\qwen38_bf16_prefill128_attention.cu" -o "%TASK_BUILD%\prefill128-attention-core.obj"
if errorlevel 1 exit /b %errorlevel%
"%TASK_NVCC%" -std=c++17 -arch=sm_120 -O3 -Xcompiler=/MT,/utf-8 -I"%TASK_ROOT%\Garnet\src\cuda" -c "%TASK_METHODS%\probe_bf16_prefill128_attention.cu" -o "%TASK_BUILD%\prefill128-attention-probe.obj"
if errorlevel 1 exit /b %errorlevel%
"%TASK_NVCC%" -std=c++17 -arch=sm_120 -Xcompiler=/MT "%TASK_BUILD%\prefill128-attention-core.obj" "%TASK_BUILD%\prefill128-attention-probe.obj" -o "%TASK_BUILD%\prefill128-attention-probe.exe"
exit /b %errorlevel%
