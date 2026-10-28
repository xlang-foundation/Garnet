"""Exclusive actual ordinary stage captures; admission oracle remains correct T32."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='q128-model-admission-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import snapshot_targets,restore_targets,candidate_bytes
    prep=M/'evidence/q128-model-admission-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='q128_model_admission_prepared_gpu_unrun' and len(p['candidate_files'])==57
    for n,h in p['files'].items():assert sha(ROOT/n)==h,('Source baseline drift',n)
    for n,h in {**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(ROOT/n)==h
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    candidate=ROOT/'build/bin/garnet-native-q128-canonical-attempt1.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';library=ROOT/'build/bin/garnet.dll'
    assert sha(candidate)==p['candidate_library_sha256'] and sha(baseline)==sha(library)==p['runtime']['garnet.dll']
    bundle=ROOT/p['source_bundle'];assert sha(bundle)==p['source_bundle_sha256'];sources=candidate_bytes(bundle,p['candidate_files']);old=snapshot_targets(ROOT/'Garnet',sources,p['candidate_base'])
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root'])
    assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();assert shutil.disk_usage(ROOT).free>=4<<30
    out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=dict(p['files'])
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    w=dict(status='running',source_before_sha256=p['files'],runtime_dependencies=p['runtime'],environment_before=p['environment'],candidate_library_sha256=sha(candidate),preparation_sha256=sha(prep),parents=p['parents'],candidate_files=p['candidate_files'],candidate_base=p['candidate_base'],cleanup_finished=False,performance_admitted=False,scope=p['scope'])
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[ROOT/n for n in {**p['audit_dependencies'],**p['checkpoints'],**p['new_files']}]+[prep,bundle,receipt]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']]
    ledger.source_identity=lambda:{'scope':'Explicit owned layer3 stage observation inventory; protected unrelated untracked bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}};save()
    try:
        from test2026.cuda.qwen38_5090.q128_canonical_source import sources as derive_sources
        observation51,expected,_=derive_sources();assert expected==sources
        from test2026.cuda.qwen38_5090.full_chunk_stage_source import sources as historical_sources
        parent,qualified51,_=historical_sources();assert qualified51==observation51 and len(parent)==50
        for n,b in parent.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        inventory.update({'Garnet/'+n:hashlib.sha256(b).hexdigest() for n,b in parent.items()});assert inventory==p['parent_installed_files'];w['parent_installed_source_sha256']={n:sha(ROOT/n) for n in inventory};save()
        code=ledger.run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','59current Q128/reference/route/source/readonly/stage/array/corruption controls plus86retained historical assertions on exact parent50; no original assertion waived',common)
        if code:raise RuntimeError('CPU stage controls failed')
        for n,b in sources.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        inventory.update({'Garnet/'+n:h for n,h in p['candidate_files'].items()});assert inventory==p['installed_files'];w['installed_source_sha256']={n:sha(ROOT/n) for n in inventory};save()
        code=ledger.run(['cmd.exe','/d','/c',str(M/'test_q128_model_metadata.bat'),str(out/'embedded-metadata.json')],out/'embedded-metadata-ledger','correctness','Actual embedded stdlib-only stage metadata/AST before GPU request',common)
        if code:raise RuntimeError('Embedded stage metadata failed')
        library.write_bytes(candidate.read_bytes());assert sha(library)==p['candidate_library_sha256']
        reports={};reads={}
        for phase,t,q_enabled in (('off32',32,0),('off128',128,0),('enabled128',128,1)):
            argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(phase+'-resources')),'--','cmd.exe','/d','/c',str(M/'test_q128_model_admission.bat'),str(out/phase),str(cache),str(t),'1',str(q_enabled)]
            code=ledger.run(argv,out/(phase+'-ledger'),'correctness','Actual exclusive full3968 ordinary original state/vector/cache/ID/repeat gates with bounded complete first128 stages/readonly inputs; defaultoff same-profile controls then enabledQ128 against correctT32 oracle, no timing admission',common+[library]);idle()
            if code:raise RuntimeError('Actual stage/control collection failed; preserve raw evidence')
            reports[phase]=json.loads((out/phase/'report.json').read_bytes());assert reports[phase]['status']=='full_chunk_stages_collected_primary_profile_exact';reads[phase]=lambda n,phase=phase:(out/phase/n).read_bytes()
            if phase=='off128':
                from test2026.cuda.qwen38_5090.q128_model_replay import controls
                result=controls(reports,reads,ROOT/'results');(out/'defaultoff-analysis.json').write_bytes(json.dumps(result,indent=2).encode())
        from test2026.cuda.qwen38_5090.q128_model_replay import replay
        result=replay(reports,reads,ROOT/'results');(out/'stage-analysis.json').write_bytes(json.dumps(result,indent=2).encode());w['status']='Q128_one_text_case_and_stages_collected_needs_independent_replay'
    except BaseException as e:w.update(status='failed',error=repr(e));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as e:w.update(baseline_source_restored=False,restoration_error=repr(e));code=1
        w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;save();paths=[n for n in out.rglob('*') if n.is_file()]+common;names={n.relative_to(ROOT).as_posix():n for n in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in names.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in names.items()},linked_parent_archives={job:row['archive_sha256'] for job,row in p['parents'].items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(names))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
