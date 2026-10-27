@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cd /d "%~1"
cl /nologo /Bv /std:c++17 /EHsc /O2 /MT /I"D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\native_b4_prefill128_candidate" /Fo"%~1\header-controls.obj" /Fe"%~1\header-controls.exe" "D:\CantorAI\5090-study-20261023\Garnet\test2026\cuda\qwen38_5090\probe_b4_prefill128_policy.cpp"
exit /b %errorlevel%
