"""CPU-only compiler identity collection after original GPU worker terminal."""
from pathlib import Path
import hashlib,json,subprocess,datetime

root=Path('D:/CantorAI/5090-study-20261023');out=root/'results/parallel-top1-toolchain-observation.json'
assert not out.exists()
assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
vcvars=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
nvcc=root/'tools/cuda13/nvidia/cu13/bin/nvcc.exe';assert vcvars.is_file() and nvcc.is_file()
batch=root/'tools/parallel-top1-toolchain-observation.bat';assert not batch.exists()
script='@echo off\r\ncall "'+str(vcvars)+'" >nul 2>&1\r\nif errorlevel 1 exit /b %errorlevel%\r\nwhere cl\r\necho VCToolsVersion=%VCToolsVersion%\r\necho WindowsSDKVersion=%WindowsSDKVersion%\r\n'
batch.write_bytes(script.encode())
r=subprocess.run(['cmd.exe','/d','/c',str(batch)],capture_output=True,text=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data={'status':'parallel_top1_toolchain_observation_failed','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'vcvars':str(vcvars),'nvcc':str(nvcc),'observer_sha256':sha(Path(__file__)),'batch_sha256':sha(batch),
 'scope':'Compiler path/version observation only; no CUDA executable, model, build/runtime mutation or benchmark.'}
if r.returncode==0:
 try:
  paths=[Path(line.strip()) for line in r.stdout.splitlines() if line.strip().lower().endswith('cl.exe')]
  assert paths and paths[0].is_file()
  data.update(status='parallel_top1_toolchain_observed',host_compiler=str(paths[0]),tools={str(p):sha(p) for p in (vcvars,nvcc,paths[0])})
 except (OSError,AssertionError) as error:data['error']=repr(error)
out.write_text(json.dumps(data,indent=2),encoding='utf8')
print(json.dumps(data,indent=2));assert data['status']=='parallel_top1_toolchain_observed'
