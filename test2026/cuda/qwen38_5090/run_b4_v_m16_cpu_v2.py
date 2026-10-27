"""Exclusive private CPU contracts for unqualified B4T4 canonical candidate."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='b4-v-m16-cpu-attempt2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from test2026.cuda.qwen38_5090.b4_v_m16_source import sources
    prep=M/'evidence/b4-v-m16-cpu-v2-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='B4T4_V_M16_CPU_prepared_unrun'
    for n,h in {**p['files'],**p['checkpoints'],**p['cuda_objects']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cache_before_sha256']
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==sha(ROOT/'results'/(job+'-evidence.zip'))
        workflow=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert workflow['exit_code']==row['exit_code'] and workflow['cleanup_finished']
    old,candidate,record=sources();assert len(old)==67 and len(candidate)==73 and record['candidate_files']==p['candidate_files']
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256']
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None and set(z.namelist())==set(candidate) and all(z.read(n)==b for n,b in candidate.items())
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();os.chdir(ROOT/'Garnet');code=1
    common=[prep,bundle]+[ROOT/n for n in p['methods']]+[ROOT/n for n in p['compile_inputs']]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
    ledger.source_identity=lambda:dict(scope='Explicit owned private B4T4 CPU contracts; unrelated protected bodies excluded',owned_source_sha256={n:sha(ROOT/n) for n in p['files']})
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],cache_before_sha256=p['cache_before_sha256'],parents=p['parents'],cleanup_finished=False,gpu_model_executed=False,production_source_or_DLL_installed=False,numerical_or_performance_admitted=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    def run(argv,name,kind,label):
        nonlocal code
        code=ledger.run(argv,out/(name+'-ledger'),kind,label,common);w[name+'_exit_code']=code;save()
        if code:raise RuntimeError(name+' failed; preserve exact methods and raw failure')
    save()
    try:
        run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_b4_v_m16_source','test2026.cuda.qwen38_5090.test_decode_fp8_sdk_control','test2026.cuda.qwen38_5090.test_prefill128_fp8_projection'],'python','correctness','Four immutable73-source/failed-lineage/corruption controls plus eight unchanged SDK replay and independent conversion controls')
        for name in ('profile','factory'):
            run(['cmd.exe','/d','/c',str(M/('build_b4_v_m16_'+name+'_cpu.bat')),p['vcvars'],str(out/name)],name,'build','Private actual C++ '+name+' compile; CPU SDK interfaces only for factory, no production build or GPU')
        run([sys.executable,str(M/'run_b4_v_m16_factory_cpu.py'),'--directory',str(out/'factory'),'--output',str(out/'factory-controls')],'factory-controls','correctness','Actual V factory;212CPU trace/config/API/state/workspace/unsupported/geometry/cache-isolation/runtime-guard faults, no SDK math')
        run(['cmd.exe','/d','/c',str(M/'build_b4_v_m16_plugin_cpu.bat'),p['vcvars'],str(out/'plugin')],'plugin','build','Actual new plugin with original generic plugin against pinned TRT and CPU factory/operator stubs: separate wire, descriptors, scales, formats, cross-rejection, initialize/clone/terminate/failures')
        w['status']='B4T4_V_M16_CPU_contracts_collected_model_unqualified';code=0
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},cuda_objects_after={n:sha(ROOT/n) for n in p['cuda_objects']},cache_after_sha256=sha(cache),cleanup_finished=True)
        w['unchanged_before_after']=all(w[a]==w[b] for a,b in [('source_before','source_after'),('runtime_before','runtime_after'),('environment_before','environment_after'),('checkpoint_before','checkpoint_after'),('cuda_objects_before','cuda_objects_after'),('cache_before_sha256','cache_after_sha256')])
        if not w['unchanged_before_after']:code=1;w['exit_code']=code
        save();files={n.relative_to(ROOT).as_posix():n for n in [*[n for n in out.rglob('*') if n.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in files.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],exit_code=code,archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
