"""Build a separate opt-in candidate; always restore baseline source and DLL."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile

ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090'
JOB='bf16-scores-tiled-native-build-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve()
    sys.path.insert(0,str(M));from run_logged import run
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/bf16-scores-tiled-native-build-preparation.json';p=json.loads(prep.read_bytes())
    for n,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/n)==h,('Before-build source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h,('SDK/environment drift',n)
    cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cache_before_sha256']
    objects={f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')}
    assert objects==p['cuda_objects_before']
    parent=ROOT/'results/native-cuda-phase-logged-bootstrap-attempt1-verified.json';q=json.loads(parent.read_bytes())
    assert sha(parent)==p['bootstrap_proof_sha256'] and q['status']=='native_cuda_logged_bootstrap_replayed'
    closed=json.loads((ROOT/'results/native-cuda-phase-logged-bootstrap-attempt1/workflow.json').read_bytes())
    assert closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
    component=ROOT/'results/bf16-scores-tiled-correctness-attempt1-verified.json';c=json.loads(component.read_bytes())
    assert c['status']=='bf16_scores_tiled_fp64_replayed' and c['archive_sha256']==p['component_archive_sha256']
    source=ROOT/'Garnet/src/cuda/qwen38_bf16_kv_low_batch_overlay.cu';old=source.read_bytes()
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and z.namelist()==['src/cuda/qwen38_bf16_kv_low_batch_overlay.cu']
        candidate=z.read(z.namelist()[0]);assert hashlib.sha256(candidate).hexdigest()==p['candidate_source_sha256']
    library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll'
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll']
    saved=ROOT/'build/bin/garnet-native-bf16-scores-tiled-attempt1.dll'
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip')
    assert not out.exists() and not archive.exists() and not saved.exists()
    idle();assert shutil.disk_usage(ROOT).free>=8<<30;out.mkdir();os.chdir(ROOT/'Garnet')
    (out/'original-overlay.cu').write_bytes(old);(out/'candidate-overlay.cu').write_bytes(candidate)
    (out/'CMakeCache-before.txt').write_bytes(cache.read_bytes())
    w={'status':'running','preparation_sha256':sha(prep),'source_before':p['base_files'],'runtime_before':p['runtime'],
       'environment_before':p['environment'],'cuda_objects_before':objects,'candidate_source_sha256':p['candidate_source_sha256'],
       'bootstrap_proof_sha256':sha(parent),'component_archive_sha256':p['component_archive_sha256'],
       'cleanup_finished':False,'performance_admitted':False,'gpu_model_executed':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save();code=1
    common=[prep,parent,component,bundle]+[ROOT/n for n in p['methods']]
    try:
        code=run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','49 exact existing B1/legacy constructor and state-transaction regressions',common)
        if code:raise RuntimeError('CPU construction regression failed before source installation')
        source.write_bytes(candidate);assert sha(source)==p['candidate_source_sha256']
        w['status']='candidate_source_installed';save()
        code=run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],out/'build-ledger','build',
                 'Existing Release/MT/sm120 build and retained SDK; one opt-in low-batch attention CUDA unit',common)
        if code:raise RuntimeError('Native candidate build failed')
        assert sha(library)!=p['runtime']['garnet.dll']
        w['candidate_library_sha256']=sha(library);save()
        after={f.relative_to(ROOT).as_posix():sha(f) for f in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')}
        assert set(after)==set(objects)
        changed=[n for n in after if after[n]!=objects[n]]
        w['cuda_objects_built']=after;w['changed_cuda_objects']=changed;save()
        assert len(changed)==1 and changed[0].endswith('/qwen38_bf16_kv_low_batch_overlay.cu.obj'),changed
        code=run([sys.executable,str(M/'probe_native_tiled_scores_selector_cpu.py'),'--root',str(ROOT),'--output',str(out/'selector-cpu')],
                 out/'selector-cpu-ledger','correctness','Actual native export exact-value/default/frozen selection across ten fresh CPU subprocesses',common+[library])
        if code:raise RuntimeError('Actual native process-frozen selector regression')
        code=run([sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(out/'tokenizer-cpu')],
                 out/'tokenizer-cpu-ledger','correctness','Actual candidate DLL955frozen native C API tokenizer regression, no GPU model',common+[library])
        if code:raise RuntimeError('Candidate tokenizer regression')
        saved.write_bytes(library.read_bytes());assert sha(saved)==w['candidate_library_sha256']
        w.update(status='native_tiled_scores_built_cpu_qualified_model_unqualified',candidate_saved=saved.relative_to(ROOT).as_posix())
    except BaseException as error:
        w.update(status='native_tiled_scores_build_failed',error=repr(error));code=code or 1
        if library.exists() and sha(library)!=p['runtime']['garnet.dll']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
    finally:
        # No trial can start here: restore the exact baseline regardless of outcome.
        library.write_bytes(baseline.read_bytes())
        if sha(source)==p['candidate_source_sha256']:source.write_bytes(old)
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['base_files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},
                 environment_after={n:sha(Path(n)) for n in p['environment']},methods_after={n:sha(ROOT/n) for n in p['methods']},
                 cache_after_sha256=sha(cache),baseline_source_restored=source.read_bytes()==old,baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
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
