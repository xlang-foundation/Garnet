"""Exclusive reversible source91 HTTP correctness collection; never a benchmark."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='b4-resident-serial-stage-profile-attempt1'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024):h.update(block)
    return h.hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import snapshot_targets,restore_targets
    from b4_http_source_bundle import load_bundle
    prep=M/'evidence/b4-resident-serial-stage-profile-preparation.json';p=json.loads(prep.read_bytes())
    assert p['status']=='B4_resident_serial_stage_profile_prepared_GPU_unrun'
    assert p['checkpoint_revision']==json.loads((ROOT/p['model_file_manifest']).read_bytes())['revision']=='482ca0f3832238542f8f5295dde86b5f22711d80'
    idle()
    for n,h in {**p['files'],**p['checkpoints'],**p['cuda_objects'],**p['model_files']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256'] and sha(ROOT/p['model_file_manifest'])==p['model_file_manifest_sha256']
    for job,row in p['parents'].items():
        q=ROOT/'results'/(job+'-verified.json');d=json.loads(q.read_bytes());assert sha(q)==row['proof_sha256'] and d['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==d['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['cleanup_finished'] and w['exit_code']==row['exit_code']
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];source=load_bundle(bundle,p['model_source_files'],p['source_files']);assert len(source)==91
    assert len(p['model_source_files'])==89 and all(p['source_files'][n]==h for n,h in p['model_source_files'].items())
    old=snapshot_targets(ROOT/'Garnet',source,p['candidate_base']);library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';saved=ROOT/p['library']
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll'] and sha(saved)==p['library_sha256']
    cache=Path(json.loads((ROOT/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root'])
    assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96<<30
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    common=[prep,bundle,ROOT/p['model_file_manifest']]+[ROOT/n for n in p['archive_files']]+[ROOT/n for n in p['checkpoints']]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
    # The ledger receives small files only. Large weight shards are streamed separately.
    assert all(n.is_file() for n in common) and not any(n.suffix=='.safetensors' for n in common)
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=p['files']
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],model_files_before=p['model_files'],cache_before_sha256=p['cache_before_sha256'],parents=p['parents'],cleanup_finished=False,performance_admitted=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit source91 resident/serial host-stage diagnosis; unchanged model/DLL, protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest',*p['http_test_modules']],out/'cpu-ledger','correctness','Six new synthetic physical-group span/ID/clock/claim controls; no model/native/GPU work.',common);w['cpu_exit_code']=code;save();assert code==0,'6 CPU controls failed'
        for n,b in source.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        library.write_bytes(saved.read_bytes());inventory=p['installed_files'];assert all(sha(ROOT/n)==h for n,h in inventory.items());w['installed_source']=inventory;w['candidate_library_sha256']=sha(library);save()
        code=ledger.run([sys.executable,str(M/'launch_b4_resident_serial_stage_cpu.py'),str(ROOT)],out/'embedded-cpu-ledger','correctness','Eighteen actual embedded pure CPU controls:12existing timing/attachment controls not previously embedded-tested, plus6new physical-group trace controls. No model/native/GPU imports.',common+[library]);w['embedded_cpu_exit_code']=code;save();assert code==0,'18 embedded controls failed'
        idle();argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'resources'),'--',sys.executable,str(M/'probe_b4_resident_serial_stage_profile.py'),'--root',str(ROOT),'--output',str(out/'native')]
        code=ledger.run(argv,out/'candidate-ledger','correctness','Separate source91 resident/serial host-wall diagnostic: same31fixed4k controls unprofiled then profiled,62realHTTPoutputs exact retainedB1, actual callbacks/dispatch and group-span replay. No large transport fixture,805repeat,CUDAkernel elapsed or clean throughput admission.',common+[library]);w['candidate_exit_code']=code;save();idle();assert code==0,'B4 HTTP collection failed'
        report=json.loads((out/'native/report.json').read_bytes());assert report['status']=='B4_resident_serial_host_stage_controls_collected_needs_independent_replay' and report['real_request_count']==62 and report['host_stages']['status']=='B4_physical_group_host_stage_intervals_replayed'
        w['status']='B4_resident_serial_stage_profile_collected_needs_independent_replay';code=0;save()
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},cuda_objects_after={n:sha(ROOT/n) for n in p['cuda_objects']},model_files_after={n:sha(ROOT/n) for n in p['model_files']},cache_after_sha256=sha(ROOT/'build/CMakeCache.txt'),baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True
        w['all_restored']=all(w[a]==w[b] for a,b in [('source_before','source_after'),('runtime_before','runtime_after'),('environment_before','environment_after'),('checkpoint_before','checkpoint_after'),('cuda_objects_before','cuda_objects_after'),('model_files_before','model_files_after'),('cache_before_sha256','cache_after_sha256')]) and w['baseline_source_restored'] and w['baseline_library_restored']
        if not w['all_restored']:code=1;w.update(status='failed',exit_code=1,error='Restoration failed; preserve unexpected edits')
        save();files={n.relative_to(ROOT).as_posix():n for n in [*[n for n in out.rglob('*') if n.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in files.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],exit_code=code,archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
