@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x86_64;%PATH%"
set "METHODS=%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090"
cd /d "%STUDY_ROOT%\build"
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O2 -arch=sm_120 -DNOMINMAX -Xcompiler /utf-8 -I"%STUDY_ROOT%\tools\TensorRT\include" -I"%STUDY_ROOT%\Garnet\src\cuda" "%METHODS%\canonical_kv_overlay_plugin_test.cu" "%METHODS%\canonical_kv_overlay_plugin.cpp" "%METHODS%\attention_kv_delta_strided_core.cu" "%STUDY_ROOT%\Garnet\src\cuda\qwen35_attention_core.cu" -L"%STUDY_ROOT%\tools\TensorRT\lib" -lnvinfer_10 -o "%STUDY_ROOT%\build\bin\garnet_canonical_kv_overlay_plugin_test.exe"
exit /b %errorlevel%
