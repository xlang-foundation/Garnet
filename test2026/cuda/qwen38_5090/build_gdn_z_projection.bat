@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "%~1"
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x86_64;%PATH%"
pushd "%~2"
"%CUDA_PATH%\bin\nvcc.exe" --version
if errorlevel 1 exit /b %errorlevel%
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -arch=sm_120 -I"%STUDY_ROOT%\Garnet\src\backends\tensorrt" -I"%STUDY_ROOT%\Garnet\src\cuda" "%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090\actual_gdn_z_projection.cu" "%STUDY_ROOT%\Garnet\src\cuda\modelopt_fp8.cu" -L"%CUDA_PATH%\lib\x64" -lcublasLt -o "%~2\control.exe"
if errorlevel 1 exit /b %errorlevel%
popd
exit /b 0
