@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
set "CUDA=%STUDY_ROOT%\tools\cuda13\nvidia\cu13"
set "METHODS=%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%CUDA%\include" "%METHODS%\probe_fp8_decode_shape_configs_v2.cpp" /Fecontract.exe /link /LIBPATH:"%CUDA%\lib\x64" cudart.lib cublasLt.lib delayimp.lib /DELAYLOAD:cudart64_13.dll /DELAYLOAD:cublasLt64_13.dll
if errorlevel 1 exit /b %errorlevel%
dumpbin /imports contract.exe > imports.txt
set "BUILD_CODE=%errorlevel%"
popd
exit /b %BUILD_CODE%
