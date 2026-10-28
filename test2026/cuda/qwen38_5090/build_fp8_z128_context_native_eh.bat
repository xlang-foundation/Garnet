@echo off
setlocal
set "STUDY_ROOT=D:\CantorAI\5090-study-20261023"
call "%~1"
if errorlevel 1 exit /b %errorlevel%
pushd "%~2"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /MT /DXLANG3_USE_SHARED /I"%STUDY_ROOT%\xlang3\sdk" "%STUDY_ROOT%\Garnet\test2026\cuda\qwen38_5090\fp8_z128_context_native_observed.cpp" /link "%STUDY_ROOT%\tools\xlang3-runtime\xlang3_runtime.lib" /OUT:"%~2\fp8_z128_context_native.dll"
if errorlevel 1 exit /b %errorlevel%
popd
exit /b 0
