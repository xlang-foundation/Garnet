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
set "CANDIDATE=%SRC%\backends\tensorrt"
set "PATH=%TRT%\bin;%CUDA%\bin;%PATH%"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /DNOMINMAX /I"%TRT%\include" /I"%CUDA%\include" /I"%CANDIDATE%" /I"%SRC%\backends\tensorrt" /I"%SRC%\cuda" "%CANDIDATE%\modelopt_fp8_plugin.cpp" "%CANDIDATE%\modelopt_nvfp4_plugin.cpp" "%CANDIDATE%\modelopt_fp8_m128_plugin.cpp" "%CANDIDATE%\modelopt_fp8_z128_plugin.cpp" "%CANDIDATE%\modelopt_fp8_z128_reconstructed_plugin.cpp" "%CANDIDATE%\modelopt_fp8_kv128_plugin.cpp" "%CANDIDATE%\modelopt_fp8_q128_plugin.cpp" "%CANDIDATE%\modelopt_fp8_o128_plugin.cpp" "%METHODS%\q128_plugin_init_stubs.cpp" "%METHODS%\probe_o128_plugin_init.cpp" /Fe"contract.exe" /link "%TRT%\lib\nvinfer_10.lib"
set "BUILD_CODE=%errorlevel%"
popd
exit /b %BUILD_CODE%
