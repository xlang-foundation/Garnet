"""Exclusive observation-only native build; preserve all baselines and failures."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='b4-v-m16-build-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_z128_factory_trace_native_build import candidate_bytes,snapshot_targets,restore_targets,validate_cuda_objects
    prep=M/'evidence/b4-v-m16-build-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='b4_v_m16_build_prepared_unrun'
    for n,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/n)==h,('Pre-build source drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256']==sha(ROOT/'results'/(job+'-evidence.zip'))
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    assert sha(ROOT/'results/b4-v-m16-build-inputs.json')==p['input_inventory_sha256']
    cache=ROOT/'build/CMakeCache.txt';assert sha(cache)==p['cache_before_sha256']
    objects={n.relative_to(ROOT).as_posix():sha(n) for n in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert objects==p['cuda_objects_before']
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];candidate=candidate_bytes(bundle,p['candidate_files']);old=snapshot_targets(ROOT/'Garnet',candidate,p['candidate_base'])
    library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';saved=ROOT/'build/bin/garnet-native-b4-v-m16-attempt1.dll'
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists() and not saved.exists()
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll'];idle();assert shutil.disk_usage(ROOT).free>=12<<30
    out.mkdir();os.chdir(ROOT/'Garnet');code=1;installed={};inventory={**p['base_files'],**p['methods']}
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['base_files'],methods=p['methods'],candidate_files=p['candidate_files'],candidate_base=p['candidate_base'],runtime_before=p['runtime'],environment_before=p['environment'],parents=p['parents'],cuda_objects_before=objects,cleanup_finished=False,gpu_model_executed=False,numerical_or_performance_admitted=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[prep,bundle]+[ROOT/n for n in p['methods']]+[ROOT/n for n in p['checkpoints']]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']]+[ROOT/'results/b4-v-m16-build-inputs.json']+[ROOT/('results/b4-v-m16-build-input-inventory/'+file) for file in ('commands.log','commands.jsonl')]+[ROOT/'results/inspect-b4-v-m16-build-inputs.py',ROOT/'results/run-b4-v-m16-build-inventory.py']
    ledger.source_identity=lambda:{'scope':'Explicit owned read-only full-chunk stage native build source inventory; unrelated untracked bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    def run(argv,name,purpose):
        nonlocal code
        code=ledger.run(argv,out/(name+'-ledger'),'build' if name=='native-build' else 'correctness',purpose,common)
        if code:raise RuntimeError(name+' failed')
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    (out/'CMakeCache-before.txt').write_bytes(cache.read_bytes());save()
    try:
        # The restored baseline intentionally lacks37 research-only files. Install
        # exact qualified parent50 before existing controls; do not alter old tests.
        from test2026.cuda.qwen38_5090.o128_canonical_source import sources as o_sources
        qualified57,qualified60,_=o_sources();assert len(qualified57)==57
        from test2026.cuda.qwen38_5090.b4_prefill128_source import sources as b4_sources
        original60,prefill61,_=b4_sources();assert original60==qualified60 and len(prefill61)==61
        from test2026.cuda.qwen38_5090.b4_decode_m16_source import sources as m16_sources
        original61,gdn67,_=m16_sources();assert original61==prefill61 and len(gdn67)==67
        from test2026.cuda.qwen38_5090.b4_v_m16_source import sources as v_sources
        original67,expected,_=v_sources();assert original67==gdn67 and len(expected)==73 and expected==candidate
        from test2026.cuda.qwen38_5090.q128_canonical_source import sources as q_sources
        observation51,qualifiedQ,_=q_sources();assert len(observation51)==51 and qualifiedQ==qualified57
        from test2026.cuda.qwen38_5090.full_chunk_stage_source import sources
        parent,qualified51,_=sources();assert len(parent)==50 and qualified51==observation51
        for n,b in parent.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        inventory.update({'Garnet/'+n:hashlib.sha256(b).hexdigest() for n,b in parent.items()})
        w['parent_installed_source']={n:sha(ROOT/n) for n in inventory};save()
        run([sys.executable,'-m','unittest',*p['test_ids']],'cpu-before','78current controls plus86historical assertions on exact qualified parent50; original tests unchanged, new73source and preserved67 composition/corruption/failedSDK-model lineage checked independently; paired1209Vprofile/212factory/V-GDN-genericcrosswire contracts required parent proof')
        for n,b in candidate.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        inventory.update({'Garnet/'+n:h for n,h in p['candidate_files'].items()});w['installed_source']={n:sha(ROOT/n) for n in inventory};save()
        from test2026.cuda.qwen38_5090.b4_v_m16_source import sources
        _,expected,_=sources();assert candidate==expected
        run(['cmd.exe','/d','/c',str(M/'build_q128_profile_cpu.bat'),p['vcvars'],str(out/'profile')],'profile','Actual header-only C++ policy/geometry/offset/precision/flag negative controls; no GPU/model or numerical qualification')
        run(['cmd.exe','/d','/c',str(M/'build_o128_profile_cpu.bat'),p['vcvars'],str(out/'o-profile')],'o-profile','Actual separate O128 header278cases with16selected/262rejected, unchanged original Q profile retained')
        run(['cmd.exe','/d','/c',str(M/'build_q128_factory_cpu.bat'),p['vcvars'],str(out/'factory')],'factory-build','Actual Q128 factory against CPU SDK fixtures, no GPU library linked')
        run([sys.executable,str(M/'run_q128_factory_cpu.py'),'--directory',str(out/'factory'),'--output',str(out/'factory-controls')],'factory-controls','182actual CPU SDK fault/trace cases including all7attrs, M16/M32/M128 API/state/workspace failures, unsupported headers and observation parity')
        run(['cmd.exe','/d','/c',str(M/'build_q128_plugin_wire_cpu.bat'),p['vcvars'],str(out/'wire')],'wire','Actual new plugin wire/descriptor/type validation and legacy KV/Z/M128/plain cross-rejection, GPU stubs unused')
        run(['cmd.exe','/d','/c',str(M/'build_o128_plugin_wire_cpu.bat'),p['vcvars'],str(out/'o-wire')],'o-wire','Actual separate O128 wire/descriptor/scale/workspace and all legacy/Q cross-rejections, shared context/GPU stubs unused')
        run(['cmd.exe','/d','/c',str(M/'build_q128_plugin_init_cpu.bat'),p['vcvars'],str(out/'init')],'init-build','Actual seven plugin initialize routes against CPU-only factory/operator stubs')
        run([sys.executable,str(M/'run_q128_plugin_init_cpu.py'),'--executable',str(out/'init/contract.exe'),'--output',str(out/'init-controls')],'init-controls','630actual initialize calls/490factory calls, all7routes including unchanged KV/GDN types, cache/clone/exception and trace policies, no GPU operators')
        run(['cmd.exe','/d','/c',str(M/'build_o128_plugin_init_cpu.bat'),p['vcvars'],str(out/'o-init')],'o-init-build','Actual eighth O128 initialize route reusing unchanged M128 CPU factory; original seven routes retained')
        run([sys.executable,str(M/'run_o128_plugin_init_cpu.py'),'--executable',str(out/'o-init/contract.exe'),'--output',str(out/'o-init-controls')],'o-init-controls','720initialize/560CPUfactory calls,8routes/10policies, cache/clone/null/standard/unknown exceptions, no GPU operators')
        run(['cmd.exe','/d','/c',str(M/'build_integrated_batch_top1_native.bat'),p['vcvars']],'native-build','Release/MT/sm120 defaultoff physicalB4T4/M16 V canonical SDK context/plugin/profile over exact67; all old plugin bodies/wires/quantizers/CUDA bytes unchanged. Faithfulconnected component baseline passed; original standaloneSDK andfullmodel failures retained.1209profile/212factory/V-GDN-genericcrosswire CPU parent required. Original three-object whitelist retained, all34hashes recorded; new factory/wholemodel/quality/performance still unqualified')
        after={n.relative_to(ROOT).as_posix():sha(n) for n in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')}
        w.update(cuda_objects_built=after,changed_cuda_objects=validate_cuda_objects(objects,after),candidate_library_sha256=sha(library));save();assert sha(library)!=p['runtime']['garnet.dll']
        for name,argv in [('selector-cpu',[sys.executable,str(M/'probe_native_tiled_scores_selector_cpu.py'),'--root',str(ROOT),'--output',str(out/'selector-cpu')]),('embedded-selector-cpu',[sys.executable,str(M/'run_tiled_scores_embedded_selector.py'),'--root',str(ROOT),'--output',str(out/'embedded-selector-cpu')]),('tokenizer-cpu',[sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(out/'tokenizer-cpu')])]:run(argv,name,'Actual candidate native selector/tokenizer CPU smoke before model execution')
        assert all(sha(ROOT/n)==h for n,h in w['installed_source'].items());saved.write_bytes(library.read_bytes());assert sha(saved)==w['candidate_library_sha256']
        w.update(status='b4_v_m16_native_built_CPU_qualified_model_unqualified',candidate_saved=saved.relative_to(ROOT).as_posix());code=0
    except BaseException as e:w.update(status='failed',error=repr(e));code=code or 1
    finally:
        if code and library.exists() and sha(library)!=p['runtime']['garnet.dll']:(out/'failed-garnet.dll').write_bytes(library.read_bytes())
        library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as e:w.update(baseline_source_restored=False,restoration_error=repr(e));code=1
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['base_files']},methods_after={n:sha(ROOT/n) for n in p['methods']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},cache_after_sha256=sha(cache),checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        save();idle();w['cleanup_finished']=True;save();paths=[n for n in out.rglob('*') if n.is_file()]+common
        if saved.exists():paths.append(saved)
        names={n.relative_to(ROOT).as_posix():n for n in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in names.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in names.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(names))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
