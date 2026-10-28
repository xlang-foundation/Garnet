"""Native selector API bridge; retain failed ctypes attempt and restore baseline."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='bf16-scores-tiled-native-api-build-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M))
    from run_logged import run
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/bf16-scores-tiled-native-api-build-preparation.json';p=json.loads(prep.read_bytes())
    for n,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/n)==h,('API build source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cache_before_sha256']
    objects={f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert objects==p['cuda_objects_before']
    failure=ROOT/'results/bf16-scores-tiled-model-admission-attempt1-verified.json';q=json.loads(failure.read_bytes())
    assert sha(failure)==p['failed_model_proof_sha256'] and q['status']=='native_tiled_scores_model_failure_preserved'
    closed=json.loads((ROOT/'results/bf16-scores-tiled-model-admission-attempt1/workflow.json').read_bytes());assert closed['exit_code']==1 and closed['cleanup_finished'] and closed['baseline_library_restored'] and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];old={n:(ROOT/'Garnet'/n).read_bytes() for n in p['candidate_files']}
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and set(z.namelist())==set(p['candidate_files']) and len(z.namelist())==len(set(z.namelist()))
        candidate={n:z.read(n) for n in z.namelist()};assert all(hashlib.sha256(v).hexdigest()==p['candidate_files'][n] for n,v in candidate.items())
    library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';saved=ROOT/'build/bin/garnet-native-bf16-scores-tiled-api-attempt1.dll'
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll']
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists() and not saved.exists()
    idle();assert shutil.disk_usage(ROOT).free>=8<<30;out.mkdir();os.chdir(ROOT/'Garnet');code=1
    for label,files in (('original',old),('candidate',candidate)):
        with zipfile.ZipFile(out/(label+'-source.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for n,raw in files.items():z.writestr(n,raw)
    (out/'CMakeCache-before.txt').write_bytes(cache.read_bytes())
    w={'status':'running','preparation_sha256':sha(prep),'source_before':p['base_files'],'runtime_before':p['runtime'],'environment_before':p['environment'],'cuda_objects_before':objects,
       'failed_model_proof_sha256':sha(failure),'cleanup_finished':False,'gpu_model_executed':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save();common=[prep,failure,bundle]+[ROOT/n for n in p['methods']]
    try:
        code=run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','49unchanged legacy/B1 CPU regressions before API build',common)
        if code:raise RuntimeError('CPU construction regression')
        for n,raw in candidate.items():(ROOT/'Garnet'/n).write_bytes(raw)
        code=run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],out/'build-ledger','build','Same tiled CUDA source; minimal native API selector bridge for embedded runtime',common)
        if code:raise RuntimeError('Native selector API build')
        w['candidate_library_sha256']=sha(library);assert w['candidate_library_sha256']!=p['runtime']['garnet.dll'];save()
        after={f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert set(after)==set(objects)
        changed=[n for n,h in objects.items() if after[n]!=h];w.update(cuda_objects_built=after,changed_cuda_objects=changed);save()
        assert all(n.endswith('/qwen38_bf16_kv_low_batch_overlay.cu.obj') for n in changed)
        for name,argv in (
            ('selector-cpu',[sys.executable,str(M/'probe_native_tiled_scores_selector_cpu.py'),'--root',str(ROOT),'--output',str(out/'selector-cpu')]),
            ('embedded-selector-cpu',[sys.executable,str(M/'run_tiled_scores_embedded_selector.py'),'--root',str(ROOT),'--output',str(out/'embedded-selector-cpu')]),
            ('tokenizer-cpu',[sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(out/'tokenizer-cpu')])):
            code=run(argv,out/(name+'-ledger'),'correctness','Actual '+name+' native compatibility before any model',common+[library])
            if code:raise RuntimeError('Native '+name+' gate')
        saved.write_bytes(library.read_bytes());assert sha(saved)==w['candidate_library_sha256'];w.update(status='native_tiled_scores_api_built_cpu_qualified_model_unqualified',candidate_saved=saved.relative_to(ROOT).as_posix())
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        if code and library.exists() and sha(library)!=p['runtime']['garnet.dll']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
        library.write_bytes(baseline.read_bytes())
        for n,raw in old.items():
            f=ROOT/'Garnet'/n
            if sha(f)==p['candidate_files'][n]:f.write_bytes(raw)
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['base_files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},
                 environment_after={n:sha(Path(n)) for n in p['environment']},methods_after={n:sha(ROOT/n) for n in p['methods']},cache_after_sha256=sha(cache),
                 baseline_source_restored=all((ROOT/'Garnet'/n).read_bytes()==raw for n,raw in old.items()),baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        save();idle();w['cleanup_finished']=True;save()
        paths=[f for f in out.rglob('*') if f.is_file()]+common
        if saved.exists():paths.append(saved)
        names={f.relative_to(ROOT).as_posix():f for f in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,f in names.items():z.write(f,n)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(f) for n,f in names.items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
