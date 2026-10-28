"""Exclusive ordinary B1/B4 coordinate sampling; full primary parity before analysis."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='key35-coordinate-trajectory-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import candidate_bytes,snapshot_targets,restore_targets
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
    from test2026.cuda.qwen38_5090.b4_k35_coordinate_trajectory import analyze
    prep=M/'evidence/b4-k35-coordinate-trajectory-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='ordinary_key35_coordinate_trajectory_prepared_GPU_unrun'
    for n,h in {**p['files'],**p['audit_dependencies'],**p['checkpoints'],**p['cuda_objects']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256']
    for job,row in p['parents'].items():
        path=ROOT/'results'/(job+'-verified.json');q=json.loads(path.read_bytes());assert sha(path)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    corpus=(M/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert hashlib.sha256(corpus).hexdigest()==p['input_corpus_sha256'];selection=load_corpus(corpus)
    bundles={mode:ROOT/p[mode+'_bundle'] for mode in ('reference','candidate')};sources={}
    for mode,path in bundles.items():assert sha(path)==p[mode+'_bundle_sha256'];sources[mode]=candidate_bytes(path,p[mode+'_source_files'])
    assert len(sources['reference'])==60 and len(sources['candidate'])==85
    old=snapshot_targets(ROOT/'Garnet',sources['candidate'],p['candidate_base']);library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll'
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll']
    for mode in sources:assert sha(ROOT/p[mode+'_library'])==p[mode+'_library_sha256']
    cache=Path(json.loads((ROOT/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root']);assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96<<30
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    common=[prep,*bundles.values()]+[ROOT/n for n in p['archive_files']]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']];assert all(n.is_file() for n in common),'Missing evidence dependencies'
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=dict(p['files'])
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],cache_before_sha256=p['cache_before_sha256'],parents=p['parents'],phases=[],cleanup_finished=False,model_or_performance_qualified=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit ordinary coordinate sampling source inventory; protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','52CPU controls including full primary/cache/ID/row/finite guards and single-carrier repeated reads/geometry/ordered timepoints/fullcache linkage/actual-vs-scheduled time distinction',common);w['cpu_exit_code']=code;save();assert code==0,'52 CPU controls failed'
        for mode in ('reference','candidate'):
            idle()
            for n,b in sources[mode].items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
            library.write_bytes((ROOT/p[mode+'_library']).read_bytes());assert sha(library)==p[mode+'_library_sha256']
            inventory={**p['files'],**{'Garnet/'+n:h for n,h in p[mode+'_source_files'].items()}};assert inventory==p[mode+'_installed_files']
            w['phases'].append(dict(mode=mode,source_installed={n:sha(ROOT/n) for n in inventory},library_sha256=sha(library),exit_code=None));save()
            code=ledger.run([sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(mode+'-resources')),'--',sys.executable,str(M/'launch_b4_k35_coordinate_trajectory.py'),str(ROOT),str(out/mode),str(cache),mode,str(out/'reference')],out/(mode+'-ledger'),'correctness','Unchanged ordinary model,3968prompt+53IDs/52consumed: two reads of existing layer35K/head0/slot4012/dim211 carrier after prefill and each decode step; all complete primary checkpoints and full cache must reproduce same-profile parents before interpreting timepoints. No graph observer/native changes or benchmark.',common+[library]);w['phases'][-1]['exit_code']=code;save();idle();assert code==0,mode+' capture failed'
        from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
        from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
        w['reference_routes_absent']=assert_disabled((out/'reference-ledger/commands.log').read_bytes());w['candidate_routes']=route_trace((out/'candidate-ledger/commands.log').read_bytes(),'off');save()
        reports={mode:json.loads((out/mode/'report.json').read_bytes()) for mode in ('reference','candidate')}
        with zipfile.ZipFile(ROOT/'results/b4-k35-cache52-attempt1-evidence.zip') as rp,zipfile.ZipFile(ROOT/'results/b4-k-m4x4-control52-attempt1-evidence.zip') as cp:
            rb='results/b4-k35-cache52-attempt1/reference/';cb='results/b4-k-m4x4-control52-attempt1/off/'
            w['trajectory_analysis']=analyze(selection,p['selected_key'],reports['reference'],reports['candidate'],json.loads(rp.read(rb+'report.json')),json.loads(cp.read(cb+'report.json')),lambda n:(out/'reference'/n).read_bytes(),lambda n:(out/'candidate'/n).read_bytes(),lambda n:rp.read(rb+n),lambda n:cp.read(cb+n))
        (out/'analysis.json').write_bytes(json.dumps(w['trajectory_analysis'],indent=2).encode());w['status']='ordinary_coordinate_trajectory_collected_no_model_admission';code=0;save()
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},cuda_objects_after={n:sha(ROOT/n) for n in p['cuda_objects']},cache_after_sha256=sha(ROOT/'build/CMakeCache.txt'),baseline_library_restored=sha(library)==p['runtime']['garnet.dll'])
        w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True
        w['all_restored']=all(w[a]==w[b] for a,b in [('source_before','source_after'),('runtime_before','runtime_after'),('environment_before','environment_after'),('checkpoint_before','checkpoint_after'),('cuda_objects_before','cuda_objects_after'),('cache_before_sha256','cache_after_sha256')]) and w['baseline_source_restored'] and w['baseline_library_restored']
        if not w['all_restored']:code=1;w.update(status='failed',exit_code=1,error='Restoration failed; preserve unexpected edits')
        save();files={n.relative_to(ROOT).as_posix():n for n in [*[n for n in out.rglob('*') if n.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(path) for n,path in files.items()},'scope':p['scope']},indent=2))
        print(json.dumps({'status':w['status'],'exit_code':code,'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)}),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
