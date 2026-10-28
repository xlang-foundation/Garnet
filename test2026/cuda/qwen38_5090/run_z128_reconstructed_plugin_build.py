"""Compile actual reconstructed plugin CPU controls, then native Garnet; restore baseline."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='z128-reconstructed-plugin-build-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_z128_factory_trace_native_build import candidate_bytes,snapshot_targets,restore_targets,validate_cuda_objects
    prep=M/'evidence/z128-reconstructed-plugin-build-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='reconstructed_Z128_plugin_build_prepared_unrun'
    for name,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/name)==h,('Pre-build source drift',name)
    for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in p['environment'].items():assert sha(Path(name))==h
    cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cache_before_sha256']
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];candidate=candidate_bytes(bundle,p['candidate_files']);old=snapshot_targets(ROOT/'Garnet',candidate,p['candidate_base'])
    library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';saved=ROOT/'build/bin/garnet-native-z128-reconstructed-plugin-attempt1.dll'
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists() and not saved.exists()
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll'];idle();assert shutil.disk_usage(ROOT).free>=12<<30
    out.mkdir();os.chdir(ROOT/'Garnet');code=1;installed={};inventory={**p['base_files'],**p['methods']}
    w={'status':'running','preparation_sha256':sha(prep),'source_before':p['base_files'],'methods':p['methods'],'candidate_files':p['candidate_files'],'candidate_base':p['candidate_base'],'runtime_before':p['runtime'],'environment_before':p['environment'],'parents':p['parents'],'cuda_objects_before':p['cuda_objects_before'],'cleanup_finished':False,'gpu_model_executed':False,'numerical_or_performance_admitted':False}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[prep,bundle,ROOT/p['proposal']]+[ROOT/name for name in p['checkpoints']]+[ROOT/name for name in p['methods']]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']]
    ledger.source_identity=lambda:{'scope':'Explicit owned reconstructed-plugin source inventory; unrelated untracked probe excluded','owned_source_sha256':{name:sha(ROOT/name) for name in inventory}}
    def run(argv,name,purpose):
        nonlocal code
        code=ledger.run(argv,out/(name+'-ledger'),'build' if name=='native-build' else 'correctness',purpose,common)
        if code:raise RuntimeError(name+' failed')
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in old.items():z.writestr(name,raw)
    (out/'CMakeCache-before.txt').write_bytes(cache.read_bytes());save()
    try:
        for name,raw in candidate.items():installed[name]=raw;(ROOT/'Garnet'/name).write_bytes(raw)
        inventory.update({'Garnet/'+name:h for name,h in p['candidate_files'].items()});w['installed_source']={name:sha(ROOT/name) for name in inventory};save()
        from test2026.cuda.qwen38_5090.z128_reconstructed_plugin_source import sources
        _,expected,_=sources();assert candidate==expected
        run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_z128_reconstructed_plugin'],'cpu-source','Six new source/route/negative controls, including unchanged86historical tests in explicit frozen parent source view. No numerical admission.')
        for name,script in [('new-wire','build_z128_reconstructed_plugin_wire_cpu.bat'),('new-init-build','build_z128_reconstructed_plugin_init_cpu.bat'),('new-profile','build_z128_reconstructed_profile_cpu.bat'),('old-wire','build_modelopt_init_trace_z128_wire_cpu.bat')]:
            run(['cmd.exe','/d','/c',str(M/script),p['vcvars'],str(out/name)],name,'Actual CPU-only plugin/header contract, no model/enqueue or GPU factory/library')
        run([sys.executable,str(M/'run_z128_reconstructed_plugin_init_cpu.py'),'--executable',str(out/'new-init-build/contract.exe'),'--output',str(out/'new-init')],'new-init','Ten fresh policy processes;450actual initialize calls/350CPU factory stubs, separate old/new routes, cache/clone/exception/default-silence,0GPU operators')
        from test2026.cuda.qwen38_5090.fp8_z128_native_build_contract import z_wire_contract
        w['old_wire_contract']=z_wire_contract(json.loads((out/'old-wire/report.json').read_bytes()));save()
        run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],'native-build','Full Release/MT/sm120 native reconstructed plugin, explicit defaultoff profile/counter and qualified API factory; no new CUDA kernel or model timing')
        after={path.relative_to(ROOT).as_posix():sha(path) for path in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')}
        w.update(cuda_objects_built=after,changed_cuda_objects=validate_cuda_objects(p['cuda_objects_before'],after),candidate_library_sha256=sha(library));save()
        assert w['candidate_library_sha256']!=p['runtime']['garnet.dll']
        for name,argv in [('selector-cpu',[sys.executable,str(M/'probe_native_tiled_scores_selector_cpu.py'),'--root',str(ROOT),'--output',str(out/'selector-cpu')]),('embedded-selector-cpu',[sys.executable,str(M/'run_tiled_scores_embedded_selector.py'),'--root',str(ROOT),'--output',str(out/'embedded-selector-cpu')]),('tokenizer-cpu',[sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(out/'tokenizer-cpu')])]:
            run(argv,name,'Actual candidate native selector/tokenizer CPU smoke contract before any model')
        assert all(sha(ROOT/name)==h for name,h in w['installed_source'].items())
        saved.write_bytes(library.read_bytes());assert sha(saved)==w['candidate_library_sha256'];w.update(status='reconstructed_Z128_plugin_native_built_CPU_qualified_model_unqualified',candidate_saved=saved.relative_to(ROOT).as_posix());code=0
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        if code and library.exists() and sha(library)!=p['runtime']['garnet.dll']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
        library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after={name:sha(ROOT/name) for name in p['base_files']},methods_after={name:sha(ROOT/name) for name in p['methods']},runtime_after={name:sha(ROOT/'build/bin'/name) for name in p['runtime']},environment_after={name:sha(Path(name)) for name in p['environment']},cache_after_sha256=sha(cache),checkpoint_after={name:sha(ROOT/name) for name in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        save();idle();w['cleanup_finished']=True;save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common
        if saved.exists():paths.append(saved)
        names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
