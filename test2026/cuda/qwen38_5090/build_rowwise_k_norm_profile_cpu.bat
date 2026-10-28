@echo off
setlocal
call "%~1" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /Bv /std:c++17 /EHsc /W4 /WX /MT /O2 /I "%~2" "%~2\probe_rowwise_k_norm_profile.cpp" /Fe:"%~3\contract.exe" /Fo:"%~3\contract.obj"
exit /b %errorlevel%
