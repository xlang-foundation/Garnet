@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
if exist "%~1" exit /b 2
mkdir "%~1"
if errorlevel 1 exit /b %errorlevel%
set "GARNET_QG_SDK_CUDA=D:\CantorAI\5090-study-20261023\tools\cuda13\nvidia\cu13"
set "PATH=%GARNET_QG_SDK_CUDA%\bin;%GARNET_QG_SDK_CUDA%\bin\x86_64;%PATH%"
pushd "%~1"
"%GARNET_QG_SDK_CUDA%\bin\nvcc.exe" -std=c++17 -arch=sm_120 -cudart shared -Xcompiler /utf-8 -Xcompiler /MT "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_qg128_sdk_control_attempt2.cpp" -L"%GARNET_QG_SDK_CUDA%\lib\x64" -lcublasLt -o qg128-sdk-control.exe
set "GARNET_QG_SDK_EXIT=%errorlevel%"
popd
exit /b %GARNET_QG_SDK_EXIT%
