@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
set "CUDA_VISIBLE_DEVICES=-1"
set "TMP=C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build"
set "TEMP=%TMP%"
cd /d "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build"
echo VCToolsVersion=%VCToolsVersion%
echo WindowsSDKVersion=%WindowsSDKVersion%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" --version
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -std=c++17 -arch=sm_120 -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -Xcompiler=/MT,/utf-8 -Xptxas=-v -I"D:\CantorAI\work\qwen38-5090\src\cuda" -I"D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090" -I"C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work" -c "D:\CantorAI\work\qwen38-5090\src\cuda\qwen38_bf16_kv_low_batch_overlay.cu" -o "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\baseline.obj"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -std=c++17 -arch=sm_120 -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -Xcompiler=/MT,/utf-8 -Xptxas=-v -I"D:\CantorAI\work\qwen38-5090\src\cuda" -I"D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090" -I"C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work" -c "D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090\bf16_low_batch_scores_tiled.cu" -o "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\candidate.obj"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -std=c++17 -arch=sm_120 -O3 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -Xcompiler=/MT,/utf-8 -Xptxas=-v -I"D:\CantorAI\work\qwen38-5090\src\cuda" -I"D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090" -I"C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work" -c "D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090\probe_bf16_scores_tiled.cu" -o "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\probe.obj"
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -std=c++17 -arch=sm_120 -ccbin "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -Xcompiler=/MT "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\baseline.obj" "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\candidate.obj" "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\probe.obj" -o "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\bf16-score-tiling-build\bf16-scores-tiled.exe"
exit /b %errorlevel%
