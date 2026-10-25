@echo off
setlocal
call "%~1"
if errorlevel 1 exit /b %errorlevel%
set "TMP=%~3"
set "TEMP=%TMP%"
cd /d "%~3"
where cl > "%~3\low-batch-plugin-host-compiler.txt"
if errorlevel 1 exit /b %errorlevel%
echo VCToolsVersion=%VCToolsVersion%
echo WindowsSDKVersion=%WindowsSDKVersion%
"%~2" --version
if errorlevel 1 exit /b %errorlevel%
set "STUDYMETHODS=%~5\Garnet\test2026\cuda\qwen38_5090"
set "STUDYSDK=%~5\tools\TensorRT"
set "STUDYCUDA=%~5\Garnet\src\cuda"
set "STUDYTRT=%~5\Garnet\src\backends\tensorrt"
set "STUDYINC=/I"%STUDYSDK%\include" /I"%~dp2..\include" /I"%STUDYCUDA%" /I"%STUDYTRT%" /I"%STUDYMETHODS%" /I"%~3""
"%~4\cl.exe" /nologo /EHsc /std:c++17 /MT /utf-8 %STUDYINC% /c "%STUDYMETHODS%\bf16_low_batch_overlay_plugin.cpp" /Fo"%~3\low-batch-plugin-wrapper.obj"
if errorlevel 1 exit /b %errorlevel%
"%~4\cl.exe" /nologo /EHsc /std:c++17 /MT /utf-8 %STUDYINC% /c "%STUDYTRT%\qwen38_bf16_kv_overlay_plugin.cpp" /Fo"%~3\low-batch-plugin-legacy-wrapper.obj"
if errorlevel 1 exit /b %errorlevel%
"%~4\cl.exe" /nologo /EHsc /std:c++17 /MT /utf-8 %STUDYINC% /c "%STUDYMETHODS%\bf16_low_batch_plugin_cpu_operators.cpp" /Fo"%~3\low-batch-plugin-cpu-operators.obj"
if errorlevel 1 exit /b %errorlevel%
"%~4\cl.exe" /nologo /EHsc /std:c++17 /MT /utf-8 %STUDYINC% /c "%STUDYMETHODS%\probe_bf16_low_batch_plugin_contract.cpp" /Fo"%~3\low-batch-plugin-cpu-probe.obj"
if errorlevel 1 exit /b %errorlevel%
link /nologo "%~3\low-batch-plugin-wrapper.obj" "%~3\low-batch-plugin-legacy-wrapper.obj" "%~3\low-batch-plugin-cpu-operators.obj" "%~3\low-batch-plugin-cpu-probe.obj" "%STUDYSDK%\lib\nvinfer_10.lib" /OUT:"%~3\bf16-low-batch-plugin-cpu.exe"
if errorlevel 1 exit /b %errorlevel%
dumpbin /imports "%~3\bf16-low-batch-plugin-cpu.exe" > "%~3\low-batch-plugin-cpu-imports.txt"
if errorlevel 1 exit /b %errorlevel%
findstr /i /c:"cudart" /c:"nvcuda" /c:"cublas" /c:"garnet.dll" "%~3\low-batch-plugin-cpu-imports.txt" >nul
if not errorlevel 1 exit /b 19
"%~2" -std=c++17 -arch=sm_120 -O3 -ccbin "%~4" -Xcompiler=/MT,/utf-8 -I"%STUDYSDK%\include" -I"%STUDYCUDA%" -I"%STUDYMETHODS%" -I"%~3" -c "%STUDYMETHODS%\bf16_kv_low_batch_candidate.cu" -o "%~3\low-batch-plugin-kernel.obj"
if errorlevel 1 exit /b %errorlevel%
"%~2" -std=c++17 -arch=sm_120 -O3 -ccbin "%~4" -Xcompiler=/MT,/utf-8 -I"%STUDYSDK%\include" -I"%STUDYCUDA%" -I"%STUDYMETHODS%" -I"%~3" -c "%STUDYMETHODS%\probe_bf16_low_batch_plugin_attention.cu" -o "%~3\low-batch-plugin-gpu-probe.obj"
if errorlevel 1 exit /b %errorlevel%
"%~2" -std=c++17 -arch=sm_120 -ccbin "%~4" -Xcompiler=/MT "%~3\low-batch-plugin-wrapper.obj" "%~3\low-batch-plugin-kernel.obj" "%~3\low-batch-plugin-gpu-probe.obj" "%STUDYSDK%\lib\nvinfer_10.lib" -o "%~3\bf16-low-batch-plugin-forwarding.exe"
if errorlevel 1 exit /b %errorlevel%
dumpbin /imports "%~3\bf16-low-batch-plugin-forwarding.exe" > "%~3\low-batch-plugin-forwarding-imports.txt"
exit /b %errorlevel%
