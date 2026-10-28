"""Exclusive full-model T128 trial; preserve and restore the complete native/Python profile."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='fp8-z128-full4k-model-admission-attempt1'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_bf16_prefill128_native_build import snapshot_targets,restore_targets
    prep=M/'evidence/fp8-z128-full4k-model-preparation.json';p=json.loads(prep.read_bytes())
    assert p['status']=='fp8_z128_full4k_model_prepared_gpu_unrun' and len(p['candidate_files'])==35
    for name,h in p['files'].items():assert sha(ROOT/name)==h,('Model baseline/source drift',name)
    for name,h in p['checkpoints'].items():assert sha(ROOT/name)==h
    for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in p['environment'].items():assert sha(Path(name))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes())
        assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        closed=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert closed['exit_code']==row['exit_code'] and closed['cleanup_finished']
    candidate=ROOT/'build/bin/garnet-native-fp8-z128-attempt1.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';library=ROOT/'build/bin/garnet.dll'
    assert sha(candidate)==p['candidate_library_sha256'] and sha(baseline)==sha(library)==p['runtime']['garnet.dll']
    bundle=ROOT/p['source_bundle'];assert sha(bundle)==p['source_bundle_sha256']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==35 and set(z.namelist())==set(p['candidate_files'])
        sources={name:z.read(name) for name in z.namelist()}
        for name,raw in sources.items():assert name.startswith(('src/','xModel/')) and hashlib.sha256(raw).hexdigest()==p['candidate_files'][name]
    old=snapshot_targets(ROOT/'Garnet',sources,p['candidate_base'])
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root'])
    assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    for phase in ('off','native'):assert not (ROOT/'results/fp8-z128-native-package-audit-attempt1'/('full4k-model-attempt1-'+phase+'-context.json')).exists()
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={path.name for path in cache.iterdir()};started=time.time();code=1;installed={}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name,raw in old.items():z.writestr(name,raw)
    w={'status':'running','source_before_sha256':p['files'],'runtime_dependencies':p['runtime'],'environment_before':p['environment'],'candidate_library_sha256':sha(candidate),'baseline_library_sha256':sha(library),
       'preparation_sha256':sha(prep),'parents':p['parents'],'candidate_files':p['candidate_files'],'candidate_base':p['candidate_base'],'cleanup_finished':False,'performance_admitted':False,'scope':p['scope']}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[ROOT/name for name in p['audit_dependencies']]+[ROOT/name for name in p['checkpoints']]+[prep,bundle,receipt]+[ROOT/name for name in p['new_files']]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']]
    inventory=dict(p['files']);ledger.source_identity=lambda:{'scope':'Explicit owned T128 model source inventory; unrelated untracked bodies excluded','owned_source_sha256':{name:sha(ROOT/name) for name in inventory}}
    save()
    try:
        for name,raw in sources.items():installed[name]=raw;(ROOT/'Garnet'/name).write_bytes(raw)
        inventory.update({'Garnet/'+name:h for name,h in p['candidate_files'].items()});assert inventory==p['installed_files']
        w['installed_source_sha256']={name:sha(ROOT/name) for name in inventory};assert w['installed_source_sha256']==p['installed_files'];save()
        code=ledger.run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','70CPU composed native28/Python7 source/routing/factory, oldB1/B4 guards, old/new MTP binding, both FP8 counters and unchanged strict retained4K oracle',common)
        if code:raise RuntimeError('CPU model/routing/replay gate failed')
        library.write_bytes(candidate.read_bytes());assert sha(library)==p['candidate_library_sha256']
        off_command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'selection-off-resources'),'--','cmd.exe','/d','/c',str(M/'test_fp8_z128_full4k_model.bat'),str(out/'selection-off'),str(cache),'0','--selection-off']
        code=ledger.run(off_command,out/'selection-off-ledger','correctness','Separate default/off target graph construction only; M128 graph counters must both be0, close model before enabled process',common+[library])
        idle()
        if code:raise RuntimeError('M128 default/off construction control failed')
        off=json.loads((out/'selection-off/report.json').read_bytes())
        from test2026.cuda.qwen38_5090.fp8_z128_full4k_model import validate_targets
        assert off['status']=='fp8_m128_selection_off_constructed' and off['fp8_m128_policy']==off['fp8_z128_policy']=='0' and off['cases']==[] and off['ordinary_observations']==[]
        w['selection_off_counts']=validate_targets(off['fp8_target_graphs'],'0');save()
        command=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'native-resources'),'--','cmd.exe','/d','/c',str(M/'test_fp8_z128_full4k_model.bat'),str(out/'native'),str(cache),'1']
        code=ledger.run(command,out/'native-ledger','correctness','Actual B1 targetT128/decodeT4/draftT16 six3968+128text/image/reset-repeat ordinary and verifiedMTP versus retained T32 complete state/raw-vector/ID oracle; no timing admission',common+[library])
        idle()
        if code:raise RuntimeError('T128 full4K model/state gate failed; preserve raw observations, no performance')
        report=json.loads((out/'native/report.json').read_bytes());assert report['local_replay']['status']=='prefill128_full4k_retained_target_state_exact'
        w['status']='fp8_z128_full4k_model_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after_sha256={name:sha(ROOT/name) for name in p['files']},runtime_dependencies_after={name:sha(ROOT/'build/bin'/name) for name in p['runtime']},environment_after={name:sha(Path(name)) for name in p['environment']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'],checkpoint_after={name:sha(ROOT/name) for name in p['checkpoints']})
        w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common;names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'linked_parent_archives':{job:row['archive_sha256'] for job,row in p['parents'].items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
