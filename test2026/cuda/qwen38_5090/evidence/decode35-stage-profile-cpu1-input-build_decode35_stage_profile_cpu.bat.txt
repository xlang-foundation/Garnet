@echo off
setlocal
call "%~1" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /W4 /WX /MT /O2 /I "%~2" "%~2\probe_decode35_stage_profile.cpp" /Fe:"%~3\contract.exe" /Fo:"%~3\contract.obj"
exit /b %errorlevel%
