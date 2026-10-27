"""Exclusive source-bound O128 HTTP quality experiment, always restore baseline."""
from pathlib import Path
import hashlib,json,os,shutil,socket,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='o128-http-quality-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import snapshot_targets,restore_targets,candidate_bytes
    prep=M/'evidence/o128-http-quality-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='O128_HTTP_quality_prepared_not_run'
    for n,h in {**p['files'],**p['new_files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(ROOT/n)==h,('Baseline drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for job,row in p['measurement_parents'].items():
        q=json.loads((ROOT/'results'/(job+'-verified.json')).read_bytes());assert sha(ROOT/'results'/(job+'-verified.json'))==row['proof_sha256'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
    candidate=ROOT/'build/bin/garnet-native-o128-canonical-attempt1.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';library=ROOT/'build/bin/garnet.dll';assert sha(candidate)==p['candidate_library_sha256'] and sha(baseline)==sha(library)==p['runtime']['garnet.dll']
    bundle=ROOT/p['source_bundle'];assert sha(bundle)==p['source_bundle_sha256'];sources=candidate_bytes(bundle,p['candidate_files']);old=snapshot_targets(ROOT/'Garnet',sources,p['candidate_base'])
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root']);assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96*1024**3
    with socket.socket() as endpoint:endpoint.bind(('127.0.0.1',18082))
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();assert shutil.disk_usage(ROOT).free>=4<<30
    out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory={**p['files'],**p['new_files']}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    w=dict(status='running',preparation_sha256=sha(prep),source_before_sha256={n:sha(ROOT/n) for n in p['files']},runtime_dependencies=p['runtime'],environment_before=p['environment'],candidate_library_sha256=sha(candidate),candidate_files=p['candidate_files'],candidate_base=p['candidate_base'],new_files=p['new_files'],measurement_parents=p['measurement_parents'],cleanup_finished=False,performance_admitted=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    common=[ROOT/n for n in {**p['audit_dependencies'],**p['new_files']}]+[prep,bundle,receipt]+[ROOT/'results'/(job+'-verified.json') for job in p['measurement_parents']]
    ledger.source_identity=lambda:{'scope':'Explicit owned O128 HTTP source inventory; protected unrelated untracked bodies excluded; no compilation','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}};save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_o128_http_admission','test2026.cuda.qwen38_5090.test_mmmu_http_quality','test2026.cuda.qwen38_5090.test_mmmu_quality_protocol'],out/'cpu-ledger','correctness','HTTP source/restoration certificate corruption controls and retained raw805 quality/protocol controls; no GPU',common)
        if code:raise RuntimeError('HTTP CPU controls failed')
        for n,b in sources.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        inventory.update({'Garnet/'+n:h for n,h in p['candidate_files'].items()});w['installed_source_sha256']={n:sha(ROOT/n) for n in inventory};save()
        library.write_bytes(candidate.read_bytes());assert sha(library)==p['candidate_library_sha256']
        argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/'resources'),'--',sys.executable,str(M/'probe_o128_http_quality.py'),'--root',str(ROOT),'--output',str(out/'native')]
        code=ledger.run(argv,out/'gpu-ledger','correctness','Actual O128 loopback MTP worker: four3968+128text/image reset controls then805frozen low-resolution adapted screening; compare every prompt/output ID against raw complete correctT32 control; no benchmark',common+[library]);idle()
        if code:raise RuntimeError('Actual O128 HTTP quality failed; preserve raw evidence')
        report=json.loads((out/'native/report.json').read_bytes());assert report['status']=='O128_HTTP_adapted805_and_four_full4k_controls_collected_needs_independent_replay' and not report['token_mismatches'];w['status']='O128_HTTP_quality_collected_needs_independent_replay'
    except BaseException as e:w.update(status='failed',error=repr(e));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as e:w.update(baseline_source_restored=False,restoration_error=repr(e));code=1
        w.update(exit_code=code,source_after_sha256={n:sha(ROOT/n) for n in p['files']},new_files_after={n:sha(ROOT/n) for n in p['new_files']},runtime_dependencies_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;save();paths=[n for n in out.rglob('*') if n.is_file()]+common;names={n.relative_to(ROOT).as_posix():n for n in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in names.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in names.items()},linked_parent_archives={job:row['archive_sha256'] for job,row in p['measurement_parents'].items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(names))),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
