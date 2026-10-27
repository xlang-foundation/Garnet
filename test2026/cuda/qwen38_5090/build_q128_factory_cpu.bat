@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "METHODS=D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090"
set "CANDIDATE=%METHODS%\native_q128_canonical_candidate"
set "FIXTURES=%METHODS%\native_q128_canonical_cpu_sdk"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%FIXTURES%" /I"%CANDIDATE%" "%CANDIDATE%\modelopt_fp8_q128_canonical.cpp" "%METHODS%\probe_q128_factory_cpu.cpp" /Fecontract.exe
if errorlevel 1 exit /b %errorlevel%
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /DCUDART_VERSION=12000 /I"%FIXTURES%" /I"%CANDIDATE%" "%CANDIDATE%\modelopt_fp8_q128_canonical.cpp" "%METHODS%\probe_q128_factory_cpu.cpp" /Feunsupported.exe
set "BUILD_CODE=%errorlevel%"
popd
exit /b %BUILD_CODE%
