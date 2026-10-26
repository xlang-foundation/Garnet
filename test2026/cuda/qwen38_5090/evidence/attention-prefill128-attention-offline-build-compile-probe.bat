@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.29
if errorlevel 1 exit /b %errorlevel%
set "TMP=C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\prefill128-attention-offline-build"
set "TEMP=C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\prefill128-attention-offline-build"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" --version
if errorlevel 1 exit /b %errorlevel%
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -std=c++17 -arch=sm_120 -O3 -c -I"D:\CantorAI\work\qwen38-5090/src/cuda" -Xcompiler=/MT,/utf-8 --ptxas-options=-v "D:\CantorAI\work\qwen38-5090\test2026\cuda\qwen38_5090/probe_bf16_prefill128_attention.cu" -o "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\prefill128-attention-offline-build\prefill128_attention_probe.obj"
if errorlevel 1 exit /b %errorlevel%
dumpbin /headers "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\prefill128-attention-offline-build\prefill128_attention_probe.obj" > "C:\Users\developer\Documents\Codex\2026-10-23\referenced-chatgpt-conversation-this-is-an\work\prefill128-attention-offline-build/probe-object-headers.txt"
exit /b %errorlevel%
