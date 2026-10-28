"""Exclusive private SDK query; no production build, matmul, model, or benchmark."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='fp8-decode-shape-configs-attempt2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/fp8-decode-shape-configs-v2-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='FP8_decode_shape_configs_v2_prepared_SDK_unrun'
    for n,h in {**p['files'],**p['checkpoints'],**p['cuda_objects']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256']
    for job,row in p['parents'].items():
        q=json.loads((ROOT/'results'/(job+'-verified.json')).read_bytes());assert sha(ROOT/'results'/(job+'-verified.json'))==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256']==sha(ROOT/'results'/(job+'-evidence.zip'))
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    idle();out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();out.mkdir();os.chdir(ROOT/'Garnet');code=1
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],cache_before_sha256=p['cache_before_sha256'],cleanup_finished=False,production_source_or_DLL_installed=False,matmul_model_or_benchmark_executed=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[prep]+[ROOT/n for n in p['methods']]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']]
    ledger.source_identity=lambda:dict(scope='Explicit owned private read-only FP8 SDK shape query; protected unrelated bodies excluded',owned_source_sha256={n:sha(ROOT/n) for n in p['files']})
    def run(argv,name,kind,purpose):
        nonlocal code
        code=ledger.run(argv,out/(name+'-ledger'),kind,purpose,common);w[name+'_exit_code']=code;save()
        if code:raise RuntimeError(name+' failed; preserve exact query and failure')
    save()
    try:
        run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_fp8_decode_shape_configs_v2'],'cpu','correctness','Four configuration-query identity/geometry/attributes/API/workspace/unsupported/no-numerical-admission controls')
        run(['cmd.exe','/d','/c',str(M/'build_fp8_decode_shape_configs_v2.bat'),p['vcvars'],str(out/'private-build')],'build','build','Private standalone MSVC/CUDA/cuBLASLt interface probe, no Garnet or production build; import inspection excludes matmul, CUDA tensor allocation, Garnet/TRT')
        run([sys.executable,str(M/'execute_fp8_decode_shape_configs_v2.py'),str(out)],'SDK','correctness','Actual live M4/M16 heuristic/all7config/M4-M16AlgoChecks4MiB for GDN_QKV/GDN_Z/FA_KV/FA_Q/FA_O geometries, no tensor allocation or matmul/model/benchmark')
        w['status']='actual_FP8_decode_shape_configuration_query_collected_no_matmul';code=0
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},cuda_objects_after={n:sha(ROOT/n) for n in p['cuda_objects']},cache_after_sha256=sha(ROOT/'build/CMakeCache.txt'),cleanup_finished=True)
        w['unchanged_before_after']=all(w[a]==w[b] for a,b in [('source_before','source_after'),('runtime_before','runtime_after'),('environment_before','environment_after'),('checkpoint_before','checkpoint_after'),('cuda_objects_before','cuda_objects_after'),('cache_before_sha256','cache_after_sha256')])
        if not w['unchanged_before_after']:code=1;w.update(status='failed',exit_code=code,error='Production source/runtime/SDK/build drift during private query')
        save();files={n.relative_to(ROOT).as_posix():n for n in [*[n for n in out.rglob('*') if n.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in files.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],exit_code=code,archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
