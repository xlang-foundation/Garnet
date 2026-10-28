@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "ROOT=D:\CantorAI\5090-study-20261023"
set "CUDA=%ROOT%\tools\cuda13\nvidia\cu13"
set "METHODS=%ROOT%\Garnet\test2026\cuda\qwen38_5090"
set "CANDIDATE=%METHODS%\native_z128_reconstruct_candidate"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%CUDA%\include" /I"%CANDIDATE%" "%CANDIDATE%\modelopt_fp8_z128_canonical.cpp" "%METHODS%\probe_z128_reconstruct_runtime_sdk.cpp" /Feconstructor.exe /link /MAP:constructor.map "%CUDA%\lib\x64\cublasLt.lib" "%CUDA%\lib\x64\cudart.lib"
if errorlevel 1 exit /b %errorlevel%
dumpbin.exe /dependents constructor.exe > dependencies.txt
set "CODE=%errorlevel%"
popd
exit /b %CODE%
