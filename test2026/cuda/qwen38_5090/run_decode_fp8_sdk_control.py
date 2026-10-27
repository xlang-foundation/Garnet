"""Exclusive untimed M4/M16 SDK control; pinned native math, no source/DLL install."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='decode-fp8-sdk-control-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
COMPILE_SOURCES=('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh')

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    from run_bf16_low_batch_native_build import idle
    from test2026.cuda.qwen38_5090.decode_fp8_sdk_control import observe,replay,parse_log
    import run_logged as ledger
    prep=M/'evidence/decode-fp8-sdk-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='actual_decode_fp8_SDK_prepared_GPU_unrun'
    for n,h in {**p['files'],**p['checkpoints'],**p['cuda_objects']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        parent=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert type(parent['exit_code']) is int and parent['exit_code']==row['exit_code'] and parent['cleanup_finished']
    assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==p['gpu_uuid']
    # The four files compiled here are NOT overridden by either immutable source60/61 bundle.
    # No overlay is needed: these exact baseline hashes were used by both saved native DLLs.
    for mode in ('reference','candidate'):
        assert sha(ROOT/p[mode+'_bundle'])==p[mode+'_bundle_sha256']
        assert not set(COMPILE_SOURCES).intersection(p[mode+'_source_files'])
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');fixture=ROOT/'results/actual-decode-fp8-m4-m16-fixture'
    assert not out.exists() and not archive.exists() and not fixture.exists();idle();out.mkdir();(out/'build').mkdir();(out/'actual').mkdir();os.chdir(ROOT/'Garnet');code=1
    common=[prep]+[ROOT/n for n in p['methods']]+[ROOT/'Garnet'/n for n in COMPILE_SOURCES]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
    ledger.source_identity=lambda:{'scope':'Explicit owned actual-decode SDK control inventory; unrelated untracked bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in p['files']}}
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],parents=p['parents'],cleanup_finished=False,gpu_model_executed=False,application_library_loaded=False,performance_admitted=False,scope=p['scope'])
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_decode_fp8_sdk_control'],out/'cpu-ledger','correctness','Four full-array CPU corruption controls; synthetic values are not hardware evidence',common)
        if code:raise RuntimeError('CPU controls failed')
        code=ledger.run([sys.executable,str(M/'prepare_decode_fp8_sdk_fixture.py')],out/'fixture-ledger','setup','Original real separate-normalization inputs, saved Garnet outputs, indexed actual checkpoint FP8 matrices/scalars; bounded CPU extraction only',common)
        if code:raise RuntimeError('Actual fixture preparation failed')
        f=json.loads((fixture/'manifest.json').read_bytes());w['fixture_manifest_sha256']=sha(fixture/'manifest.json');w['fixture_before']={n:sha(fixture/n) for n in f['files']};assert w['fixture_before']=={n:v['sha256'] for n,v in f['files'].items()};save()
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_decode_fp8_sdk_control.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Private explicit sm120 CUDA SDK executable; no production CMake/native DLL build or mutation',common)
        if code:raise RuntimeError('SDK executable build failed')
        exe=out/'build/control.exe';w['executable_sha256']=sha(exe);save();os.environ['PATH']=str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+os.pathsep+os.environ['PATH'];idle()
        command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'resources'),'--',str(exe),str(fixture),str(out/'actual')]
        code=ledger.run(command,out/'sdk-ledger','correctness','Actual fourM4/defaultM16/M4-algorithm-at-M16, two repeats, seven config attributes, real AlgoCheck and original4MiB; unsupported fails without fallback',common+[exe,fixture/'manifest.json']);w['sdk_exit_code']=code;save();idle()
        if code:raise RuntimeError('SDK capture failed; all partial outputs and unsupported checks retained')
        log=(out/'sdk-ledger/commands.log').read_bytes();w['algorithm_replay']=parse_log(log);save()
        result=observe(lambda n:(fixture/n).read_bytes(),lambda n:(out/'actual'/n).read_bytes());(out/'observation.json').write_bytes(json.dumps(result,indent=2).encode());w['observation']=result;save()
        w['strict_replay']=replay(lambda n:(fixture/n).read_bytes(),lambda n:(out/'actual'/n).read_bytes());w['status']='actual_decode_fp8_SDK_control_collected_needs_independent_replay';code=0
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},cuda_objects_after={n:sha(ROOT/n) for n in p['cuda_objects']},cleanup_finished=True)
        if (fixture/'manifest.json').exists():
            f=json.loads((fixture/'manifest.json').read_bytes());w['fixture_after']={n:sha(fixture/n) for n in f['files']}
        w['unchanged_before_after']=all(w[a]==w[b] for a,b in [('source_before','source_after'),('runtime_before','runtime_after'),('environment_before','environment_after'),('checkpoint_before','checkpoint_after'),('cuda_objects_before','cuda_objects_after')]);save()
        if not w['unchanged_before_after']:code=1;w['exit_code']=code;save()
        paths=[n for n in out.rglob('*') if n.is_file()]+common
        if (fixture/'manifest.json').exists():paths+=[fixture/'manifest.json']+[fixture/n for n,v in f['files'].items() if v['archive']]
        files={n.relative_to(ROOT).as_posix():n for n in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in files.items()},external_weights={n:v for n,v in f['files'].items() if not v['archive']} if (fixture/'manifest.json').exists() else {},linked_parent_archives=p['parents'],scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],exit_code=code,archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
