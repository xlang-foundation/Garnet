@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
if exist "%~2" exit /b 2
mkdir "%~2"
if errorlevel 1 exit /b %errorlevel%
pushd "%~2"
cl.exe /nologo /O2 /MT /std:c++17 /EHsc /I"D:\CantorAI\5090-study-20261023\Garnet\src\backends\tensorrt" "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_fa3_projection_profile_cpu.cpp" /Fe"contract.exe"
if errorlevel 1 exit /b %errorlevel%
contract.exe > report.json
if errorlevel 1 exit /b %errorlevel%
type report.json
popd
exit /b 0
