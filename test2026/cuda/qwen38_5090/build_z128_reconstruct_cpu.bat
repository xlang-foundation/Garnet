@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "METHODS=D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090"
set "CANDIDATE=%METHODS%\native_z128_reconstruct_candidate"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%CANDIDATE%\cpu_sdk" /I"%CANDIDATE%" "%CANDIDATE%\modelopt_fp8_z128_canonical.cpp" "%METHODS%\probe_z128_reconstruct_cpu.cpp" /Fecontract.exe
if errorlevel 1 exit /b %errorlevel%
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /DCUDART_VERSION=12000 /I"%CANDIDATE%\cpu_sdk" /I"%CANDIDATE%" "%CANDIDATE%\modelopt_fp8_z128_canonical.cpp" "%METHODS%\probe_z128_reconstruct_cpu.cpp" /Feunsupported.exe
set "CODE=%errorlevel%"
popd
exit /b %CODE%
