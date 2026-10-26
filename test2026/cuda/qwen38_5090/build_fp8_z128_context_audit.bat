@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "%~1"
if errorlevel 1 exit /b %errorlevel%
set "CUDA_PATH=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "METHODS=%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\bin\x86_64;%PATH%"
pushd "%~2"
"%CUDA_PATH%\bin\nvcc.exe" --version
if errorlevel 1 exit /b %errorlevel%
"%CUDA_PATH%\bin\nvcc.exe" -std=c++17 -shared -arch=sm_120 -Xcompiler /MT -I"%METHODS%\native_fp8_z128_candidate" "%METHODS%\fp8_z128_context_audit.cpp" -L"%CUDA_PATH%\lib\x64" -lcublasLt -o "%~2\context-audit.dll"
if errorlevel 1 exit /b %errorlevel%
popd
exit /b 0
