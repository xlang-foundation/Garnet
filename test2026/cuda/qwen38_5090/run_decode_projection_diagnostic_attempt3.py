"""Exclusive short mixed B4-vs-B1 numerical admission; no timing promotion."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='decode-projection-diagnostic-attempt3';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import snapshot_targets,restore_targets,candidate_bytes
    prep=M/'evidence/decode-projection-attempt3-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='decode_projection_prepared_GPU_unrun'
    for n,h in {**p['files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert type(w['exit_code']) is int and w['exit_code']==row['exit_code'] and w['exit_code'] in (0,1) and w['cleanup_finished']
    bundles={mode:ROOT/p[mode+'_bundle'] for mode in ('reference','candidate')}
    for mode,bundle in bundles.items():assert sha(bundle)==p[mode+'_bundle_sha256']
    sources={mode:candidate_bytes(bundle,p[mode+'_source_files']) for mode,bundle in bundles.items()};assert len(sources['reference'])==60 and len(sources['candidate'])==61
    old=snapshot_targets(ROOT/'Garnet',sources['candidate'],p['candidate_base']);baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';library=ROOT/'build/bin/garnet.dll'
    assert sha(baseline)==sha(library)==p['runtime']['garnet.dll']
    saved={mode:ROOT/p[mode+'_library'] for mode in sources}
    for mode,path in saved.items():assert sha(path)==p[mode+'_library_sha256']
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root']);assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96<<30
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();assert shutil.disk_usage(ROOT).free>=4<<30
    out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=dict(p['files'])
    common=[prep,*bundles.values()]+[ROOT/n for n in {**p['new_files'],**p['checkpoints'],**p['capture_inputs']}]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],candidate_base=p['candidate_base'],parents=p['parents'],cleanup_finished=False,performance_admitted=False,full4k_MTP_805_qualified=False,phases=[])
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit B4-vs-four-B1 short mixed numerical admission; protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_decode_projection_diagnostic'],out/'cpu-ledger','correctness','Strict diagnostic full-array finite/geometry/row/active-inactive/signedzero/input-token/repeat corruption controls; original model admission gates unchanged',common);assert code==0,'CPU gate controls failed'
        for mode in ('reference','candidate'):
            idle()
            for n,b in sources[mode].items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
            inventory={**p['files'],**{'Garnet/'+n:h for n,h in p[mode+'_source_files'].items()}};assert inventory==p[mode+'_installed_files']
            library.write_bytes(saved[mode].read_bytes());assert sha(library)==p[mode+'_library_sha256'];w['phases'].append(dict(mode=mode,source_installed={n:sha(ROOT/n) for n in inventory},library_sha256=sha(library),exit_code=None));save()
            argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(mode+'-resources')),'--',sys.executable,str(M/'launch_decode_projection_diagnostic_attempt3.py'),str(ROOT),str(out/mode),str(cache),mode]
            code=ledger.run(argv,out/(mode+'-ledger'),'correctness','Actual existing one-layer GDN chunk with shared compressed weights, first5agreed decode input tokens, T4 B1/M4 versus B4/M16. Zero recurrent/conv states; capture full retained QKV/Z/beta/a and separate normalization arrays twice. No new native source or math policy, fullmodel or timing promotion.',common+[library]);w['phases'][-1]['exit_code']=code;save();idle()
            if code:raise RuntimeError(mode+' diagnostic capture failed; preserve raw arrays and errors')
            assert (out/mode/'cublaslt.log').is_file() and (out/mode/'cublaslt.log').stat().st_size>0,'Requested actual SDK selection trace missing'
        from test2026.cuda.qwen38_5090.decode_projection_diagnostic import replay
        reference=json.loads((out/'reference/report.json').read_bytes());candidate=json.loads((out/'candidate/report.json').read_bytes())
        result=replay(reference,candidate,lambda n:(out/'reference'/n).read_bytes(),lambda n:(out/'candidate'/n).read_bytes(),p['token_inputs']);(out/'analysis.json').write_bytes(json.dumps(result,indent=2).encode())
        w['status']='actual_decode_projection_diagnostic_collected_no_model_or_performance_admission';code=0
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll']);w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;save()
        files={n.relative_to(ROOT).as_posix():n for n in [*[n for n in out.rglob('*') if n.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in files.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],exit_code=code,archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
