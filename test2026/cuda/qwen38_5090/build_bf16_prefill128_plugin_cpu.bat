@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "TRT=%STUDY_ROOT%\tools\TensorRT"
set "CUDA=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "SRC=%STUDY_ROOT%\Garnet\src"
set "METHODS=%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090"
set "PATH=%TRT%\bin;%CUDA%\bin;%PATH%"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /DNOMINMAX /I"%TRT%\include" /I"%CUDA%\include" /I"%SRC%\backends\tensorrt" /I"%SRC%\cuda" "%SRC%\backends\tensorrt\qwen38_bf16_prefill128_kv_overlay_plugin.cpp" "%SRC%\backends\tensorrt\qwen38_bf16_kv_overlay_plugin.cpp" "%SRC%\backends\tensorrt\qwen38_bf16_low_batch_kv_overlay_plugin.cpp" "%METHODS%\prefill128_plugin_cpu_operators.cpp" "%METHODS%\probe_bf16_prefill128_plugin_contract.cpp" /Fe"contract.exe" /link "%TRT%\lib\nvinfer_10.lib"
if errorlevel 1 exit /b %errorlevel%
contract.exe > report.json
if errorlevel 1 exit /b %errorlevel%
type report.json
popd
exit /b 0
