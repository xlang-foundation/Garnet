"""Compile actual native method bodies against simulated CPU resource APIs.

Usage: python test_host_embedding_cpu_extracted.py <existing-work-directory>
No CUDA runtime, GPU use, full native ABI or real driver-failure qualification.
"""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
root=Path(__file__).resolve().parents[3];work=Path(sys.argv[1]).resolve(strict=True)
source=root/'src/backends/tensorrt/cuda_tensor_io.cpp';raw=source.read_bytes();text=raw.decode().replace('\r\n','\n')
methods=text[text.index('void CUDATensorIO::EnableHostEmbedding('):text.index('X::Value CUDATensorIO::Zeros(')]
assert methods.count('CUDATensorIO::')==2
harness=Path(__file__).with_name('host_embedding_cpu_harness.cpp');template=harness.read_text(encoding='utf-8-sig')
assert template.count('// INSERT_ACTUAL_METHODS')==1
cpp=work/'host-embedding-cpu-test.cpp';exe=work/'host-embedding-cpu-test.exe';obj=work/'host-embedding-cpu-test.obj'
cpp.write_text(template.replace('// INSERT_ACTUAL_METHODS',methods),encoding='utf-8')
vcvars=Path('C:/Program Files/Microsoft Visual Studio/18/Community/VC/Auxiliary/Build/vcvars64.bat');assert vcvars.is_file()
launcher=work/'build-host-embedding-cpu-test.bat'
launcher.write_text('@echo off\ncall "'+str(vcvars)+'" >nul\nif errorlevel 1 exit /b %errorlevel%\ncl /nologo /std:c++17 /EHsc /W4 "'+str(cpp)+'" /Fo"'+str(obj)+'" /Fe"'+str(exe)+'"\nexit /b %errorlevel%\n',encoding='utf-8')
build=subprocess.run(['cmd.exe','/d','/c',str(launcher)],capture_output=True,text=True)
run=subprocess.run([str(exe)],capture_output=True,text=True) if build.returncode==0 else None
sha=lambda b:hashlib.sha256(b).hexdigest()
result={'status':'cpu_source_extracted_host_embedding_passed' if run and run.returncode==0 else 'cpu_source_extracted_host_embedding_failed',
    'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'native_source_sha256':sha(raw),'normalized_extracted_methods_sha256':sha(methods.encode()),
    'harness_sha256':sha(harness.read_bytes()),'generated_cpu_source_sha256':sha(cpp.read_bytes()),
    'build':{'exit_code':build.returncode,'stdout':build.stdout,'stderr':build.stderr},
    'run':{'exit_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr} if run else None,
    'binary_sha256':sha(exe.read_bytes()) if run else None,'gpu_used':False,'real_cuda_failure_injection':False,'performance_admitted':False,
    'scope':'Actual extracted EnableHostEmbedding/HostEmbeddingRow methods with simulated mapper/allocator/device sync. Exhaustive bit-placement and lifecycle atomicity policy only; not native ABI, real CUDA failure/lifetime, model or speed qualification.'}
(work/'host-embedding-cpu-test-result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2));raise SystemExit(0 if run and run.returncode==0 else 1)
