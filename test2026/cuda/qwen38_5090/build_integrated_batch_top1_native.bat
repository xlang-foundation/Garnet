@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "%~1"
if errorlevel 1 exit /b %errorlevel%
where cl
if errorlevel 1 exit /b %errorlevel%
echo VCToolsVersion=%VCToolsVersion%
echo WindowsSDKVersion=%WindowsSDKVersion%
set "CUDA_PATH=%STUDY_ROOT%/tools/cuda13/nvidia/cu13"
set "PATH=%STUDY_ROOT%\tools\python\ninja\data\bin;%CUDA_PATH%\bin;%CUDA_PATH%\bin\x64;%CUDA_PATH%\bin\x86_64;%STUDY_ROOT%\tools\TensorRT\bin;%PATH%"
"%STUDY_ROOT%\tools\python\cmake\data\bin\cmake.exe" --version
if errorlevel 1 exit /b %errorlevel%
"%STUDY_ROOT%\tools\python\cmake\data\bin\cmake.exe" -S "%STUDY_ROOT%/Garnet/test2026/cuda/qwen38_5090" -B "%STUDY_ROOT%/build" -DGARNET_STAGE_VENDOR_DLLS=OFF -DGARNET_QWEN35_WHOLE_PROGRAM_CUDA=OFF
if errorlevel 1 exit /b %errorlevel%
"%STUDY_ROOT%\tools\python\cmake\data\bin\cmake.exe" --build "%STUDY_ROOT%/build" --target garnet garnet_batch_top1_policy_test -j 2
if errorlevel 1 exit /b %errorlevel%
dumpbin /imports "%STUDY_ROOT%\build\bin\garnet_batch_top1_policy_test.exe"
exit /b %errorlevel%
