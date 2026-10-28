"""Compare actual embedded stdio under ordinary and profiled CPU process trees."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def driver(out):
    assert out.resolve().is_relative_to((ROOT/'results').resolve()) and not out.exists();out.mkdir()
    launcher=out/'cpu-native-target.bat'
    source='@echo off\nsetlocal\nset GARNET_RUNTIME_ROOT='+str(ROOT/'build/bin')+'\nset PATH=%GARNET_RUNTIME_ROOT%;%PATH%\ncd /d '+str(ROOT/'Garnet')+'\n%GARNET_RUNTIME_ROOT%\\xlang3.exe '+str(M/'probe_native_stdio_cpu.py')+' "%~1"\nexit /b %errorlevel%\n'
    launcher.write_bytes(source.encode());command=['cmd.exe','/d','/c',str(launcher),str(out/'embedded')]
    child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    # Small bounded response; communicate drains both pipes and owns this child.
    stdout,stderr=child.communicate(b'{"probe":"native.stdin","cpu_only":true}\n',timeout=180)
    (out/'stdout.bin').write_bytes(stdout);(out/'stderr.bin').write_bytes(stderr)
    target=out/'embedded/report.json';report=json.loads(target.read_bytes()) if target.exists() else None
    r={'status':'embedded_cpu_stdio_driver_collected','argv':command,'child_exit_code':child.returncode,
       'target':report,'stdout_bytes':len(stdout),'stderr_bytes':len(stderr),'actual_native_model_executed':False}
    (out/'driver-report.json').write_bytes(json.dumps(r,indent=2).encode())
    assert type(child.returncode) is int and child.returncode==0 and report['status']=='cpu_stdio_observations_collected'
    assert json.loads((out/'embedded/file-output.json').read_bytes())=={'probe':'owned.file','cpu_only':True}
    return 0

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--driver',type=Path);args=a.parse_args()
    assert Path(__file__).resolve().parent==M.resolve()
    if args.driver is not None:return driver(args.driver)
    sys.path.insert(0,str(ROOT/'Garnet'))
    from test2026.cuda.qwen38_5090.native_cuda_phase_study import profiler
    from test2026.cuda.qwen38_5090.run_logged import run
    out=ROOT/'results/native-embedded-stdio-cpu-diagnosis-attempt1';assert not out.exists();out.mkdir();os.chdir(ROOT/'Garnet')
    tool=profiler(ROOT);nsys=ROOT/tool['path'];runtime=ROOT/'build/bin'
    names=('garnet.dll','xlang3.exe','xlang3_runtime.dll','nvinfer_10.dll','cudart64_13.dll','cublasLt64_13.dll')
    before={name:sha(runtime/name) for name in names};sources={name:sha(M/name) for name in ('probe_native_stdio_cpu.py','run_native_profiled_stdio_cpu.py')}
    report={'status':'running','profiler':tool,'runtime_before':before,'sources':sources,'cases':[],
        'scope':'Actual CPU-only embedded stdio child with the same nested cmd/driver/Nsight process tree as the failed native profiler. No Garnet import/model/CUDA kernel call. All observed transport failures retained; no serving metrics.'}
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save();code=1
    try:
        for label in ('ordinary','profiled'):
            command=[sys.executable,str(M/'run_native_profiled_stdio_cpu.py'),'--driver',str(out/label)]
            if label=='profiled':
                application=subprocess.list2cmdline(command)
                command=[str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
                    '--force-overwrite=false','--output='+str(out/'cpu-trace'),'cmd.exe','/d','/c',application]
            code=run(command,out/(label+'-ledger'),'correctness','Actual CPU-only embedded stdio observation: '+label,
                [M/'probe_native_stdio_cpu.py',Path(__file__),runtime/'xlang3.exe',nsys])
            if code:raise RuntimeError('CPU stdio observation driver failed: '+label)
            observed=json.loads((out/label/'driver-report.json').read_bytes());report['cases'].append(dict(label=label,**observed));save()
        report['status']='actual_embedded_ordinary_and_profiled_cpu_stdio_observed';code=0
    except BaseException as error:report.update(status='failed',error=repr(error));code=code or 1
    finally:
        report.update(exit_code=code,runtime_after={name:sha(runtime/name) for name in names},profiler_after=profiler(ROOT));save()
        assert report['runtime_after']==before and report['profiler_after']==tool and sources=={name:sha(M/name) for name in sources}
    print(json.dumps({'status':report['status'],'exit_code':code,'cases':[{k:c[k] for k in ('label','child_exit_code','target')} for c in report['cases']]}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
