@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
set "TMP=%~3"
set "TEMP=%TMP%"
cd /d "%~3"
where cl > "%~3\low-batch-commit-host-compiler.txt"
if errorlevel 1 exit /b %errorlevel%
echo VCToolsVersion=%VCToolsVersion%
echo WindowsSDKVersion=%WindowsSDKVersion%
"%~2" --version
if errorlevel 1 exit /b %errorlevel%
"%~4\cl.exe" /nologo /EHsc /std:c++17 /MT /utf-8 /I"%~dp2..\include" /I"D:\CantorAI\5090-study-20261023\Garnet\src\cuda" "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_bf16_low_batch_commit_policy.cpp" /Fo"%~3\low-batch-commit-policy.obj" /Fe"%~3\low-batch-commit-policy.exe"
if errorlevel 1 exit /b %errorlevel%
dumpbin /imports "%~3\low-batch-commit-policy.exe" > "%~3\low-batch-commit-policy-imports.txt"
if errorlevel 1 exit /b %errorlevel%
findstr /i /c:"cudart" /c:"nvcuda" /c:"cublas" /c:"nvinfer" /c:"garnet.dll" "%~3\low-batch-commit-policy-imports.txt" >nul
if not errorlevel 1 exit /b 19
"%~3\low-batch-commit-policy.exe"
if errorlevel 1 exit /b %errorlevel%
"%~2" -std=c++17 -arch=sm_120 -O3 -ccbin "%~4" -Xcompiler=/MT,/utf-8 "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\bf16_low_batch_commit_candidate.cu" "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_bf16_low_batch_commit.cu" -I"D:\CantorAI\5090-study-20261023\Garnet\src\cuda" -I"%~3" -o "%~3\bf16-low-batch-commit.exe"
exit /b %errorlevel%
