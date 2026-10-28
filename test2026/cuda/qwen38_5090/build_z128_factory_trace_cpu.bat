@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "METHODS=D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090"
set "CANDIDATE=%METHODS%\native_z128_factory_trace_candidate"
pushd "%~2"
copy /b "%CANDIDATE%\original-modelopt_fp8_z128_canonical.cpp.txt" original.cpp >nul
if errorlevel 1 exit /b %errorlevel%
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%CANDIDATE%\cpu_sdk" /I"%CANDIDATE%" original.cpp "%METHODS%\probe_z128_factory_trace_cpu.cpp" /Feoriginal.exe
if errorlevel 1 exit /b %errorlevel%
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%CANDIDATE%\cpu_sdk" /I"%CANDIDATE%" "%CANDIDATE%\modelopt_fp8_z128_canonical.cpp" "%METHODS%\probe_z128_factory_trace_cpu.cpp" /Fetraced.exe
set "BUILD_CODE=%errorlevel%"
popd
exit /b %BUILD_CODE%
