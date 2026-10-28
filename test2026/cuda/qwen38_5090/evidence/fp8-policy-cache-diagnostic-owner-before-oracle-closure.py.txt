"""Preserved failed-build DLL: API/cache correctness diagnosis only, never promotion."""
from pathlib import Path
import hashlib,json,os,shutil,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='fp8-policy-cache-production-diagnostic-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_z128_factory_trace_native_build import candidate_bytes,snapshot_targets,restore_targets
    prep=M/'evidence/fp8-policy-cache-production-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='FP8_policy_cache_exploratory_diagnostic_prepared' and p['native_build_artifact_identity_gate_passed'] is False and p['exploratory_only'] is True
    for n,h in {**p['files'],**p['new_files'],**p['checkpoints']}.items():assert sha(ROOT/n)==h,('Source/method/checkpoint drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for name,row in p['parents'].items():
        proof=ROOT/'results'/(name+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256']==sha(ROOT/'results'/(name+'-evidence.zip'));w=json.loads((ROOT/'results'/name/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    failed=ROOT/'results/fp8-policy-cache-build-attempt1/failed-garnet.dll';assert sha(failed)==p['candidate_library_sha256'];bundle=ROOT/p['source_bundle'];assert sha(bundle)==p['source_bundle_sha256'];sources=candidate_bytes(bundle,p['candidate_files']);old=snapshot_targets(ROOT/'Garnet',sources,p['candidate_base'])
    library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';assert sha(baseline)==sha(library)==p['runtime']['garnet.dll'];out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();assert shutil.disk_usage(ROOT).free>=4<<30;out.mkdir();os.chdir(ROOT/'Garnet');installed={};code=1
    objects={n.relative_to(ROOT).as_posix():sha(n) for n in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert objects==p['current_cuda_objects']
    w=dict(status='running',source_before_sha256=p['files'],runtime_dependencies=p['runtime'],environment_before=p['environment'],candidate_library_sha256=sha(failed),candidate_files=p['candidate_files'],candidate_base=p['candidate_base'],parents=p['parents'],preparation_sha256=sha(prep),native_build_artifact_identity_gate_passed=False,exploratory_only=True,model_or_performance_admitted=False,cleanup_finished=False,CUDA_objects_before=objects,scope=p['scope'])
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[prep,bundle]+[ROOT/n for n in {**p['new_files'],**p['checkpoints']}]+[ROOT/'results'/(name+'-verified.json') for name in p['parents']]
    inventory=dict(p['files']);ledger.source_identity=lambda:{'scope':'Explicit exploratory failed-build cache/API diagnostic source inventory; protected unrelated bodies excluded; no promotion/performance','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_fp8_policy_cache_production_contract'],out/'cpu-ledger','correctness','Three cache matrix/collision/alias/cache-preservation controls; native artifact gate remains failed',common)
        if code:raise RuntimeError('CPU cache controls failed')
        for n,b in sources.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        inventory.update({'Garnet/'+n:h for n,h in p['candidate_files'].items()});assert inventory==p['installed_files'];w['installed_source_sha256']={n:sha(ROOT/n) for n in inventory};save();library.write_bytes(failed.read_bytes());assert sha(library)==p['candidate_library_sha256']
        for name,argv in [('selector-cpu',[sys.executable,str(M/'probe_native_tiled_scores_selector_cpu.py'),'--root',str(ROOT),'--output',str(out/'selector-cpu')]),('embedded-selector-cpu',[sys.executable,str(M/'run_tiled_scores_embedded_selector.py'),'--root',str(ROOT),'--output',str(out/'embedded-selector-cpu')]),('tokenizer-cpu',[sys.executable,str(M/'probe_native_tokenizer_frozen_cpu.py'),'--study',str(ROOT),'--oracle',str(ROOT/'results/tokenizer-nfc-cpu-oracles.zip'),'--output',str(out/'tokenizer-cpu')])]:
            code=ledger.run(argv,out/(name+'-ledger'),'correctness','Actual preserved unqualified DLL native API smoke: '+name+'; no model/performance admission',common+[library])
            if code:raise RuntimeError(name+' failed')
        argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'resources'),'--',sys.executable,str(M/'run_fp8_policy_cache_production_cases.py'),str(ROOT),str(out/'native')]
        code=ledger.run(argv,out/'gpu-ledger','correctness','41actual production load_model cache/alias/collision/invalid-policy cases and tiny addition GPU output; exploratory only, no VLM decode/quality/performance',common+[library]);idle()
        if code:raise RuntimeError('Actual production cache diagnostic failed; preserve all native errors')
        report=json.loads((out/'native/report.json').read_bytes());from test2026.cuda.qwen38_5090.fp8_policy_cache_production_contract import replay
        result=replay(report,p['candidate_library_sha256']);assert result==report['local_replay'];(out/'cache-analysis.json').write_bytes(json.dumps(result,indent=2).encode());w['status']='actual_production_cache_diagnostic_collected_needs_replay'
    except BaseException as e:w.update(status='failed',error=repr(e));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as e:w.update(baseline_source_restored=False,restoration_error=repr(e));code=1
        after={n.relative_to(ROOT).as_posix():sha(n) for n in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert after==objects
        w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'],CUDA_objects_after=after,cleanup_finished=True);save();paths=[n for n in out.rglob('*') if n.is_file()]+common;names={n.relative_to(ROOT).as_posix():n for n in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in names.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in names.items()},linked_parent_archives={name:row['archive_sha256'] for name,row in p['parents'].items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(names),exploratory_only=True,native_build_artifact_identity_gate_passed=False)),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
