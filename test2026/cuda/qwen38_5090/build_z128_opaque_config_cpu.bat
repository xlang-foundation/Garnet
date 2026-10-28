@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
set "ROOT=D:\CantorAI\5090-study-20261023"
set "CUDA=%ROOT%\tools\cuda13\nvidia\cu13"
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"%CUDA%\include" "%ROOT%\Garnet\test2026\cuda\qwen38_5090\probe_z128_opaque_config_cpu.cpp" /Feconfig-query.exe /link "%CUDA%\lib\x64\cublasLt.lib"
set "CODE=%errorlevel%"
popd
exit /b %CODE%
