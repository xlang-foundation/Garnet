@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x86_64;%PATH%"
cd /d "%STUDY_ROOT%\build"
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -O2 -arch=sm_120 -I"%STUDY_ROOT%\Garnet\src\cuda" "%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090\attention_b4_t4_test.cu" "%STUDY_ROOT%\Garnet\src\cuda\qwen35_attention_core.cu" -o "%STUDY_ROOT%\build\bin\garnet_attention_b4_t4_test.exe"
exit /b %errorlevel%
