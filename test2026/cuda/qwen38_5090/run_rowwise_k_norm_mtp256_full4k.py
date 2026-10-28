"""Exclusive source89 rowwise K MTP full4K128 target exactness; no timing."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='rowwise-k-norm-mtp256-full4k-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import candidate_bytes,snapshot_targets,restore_targets
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
    from test2026.cuda.qwen38_5090.rowwise_k_norm_analysis import norm_trace
    from test2026.cuda.qwen38_5090.b4_full4k_exact_gate import observe
    from test2026.cuda.qwen38_5090.b4_mtp256_full4k_gate import admit as mtp_admit
    from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
    from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
    prep=M/'evidence/rowwise-k-norm-mtp256-full4k-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='rowwise_K_norm_mtp256_full4k_prepared_GPU_unrun'
    for n,h in {**p['files'],**p['checkpoints'],**p['cuda_objects']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256']
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        closed=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert closed['exit_code']==row['exit_code'] and closed['cleanup_finished']
    qualified=json.loads((ROOT/'results/rowwise-k-norm-full4k-attempt1-verified.json').read_bytes());assert qualified['exit_code']==0 and qualified['ordinary_full4K128_case_gate_passed'] and qualified['strict_admission']['all_states_and_vectors_exact'] and qualified['strict_admission']['all_128_IDs_exact']
    corpus=(M/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert hashlib.sha256(corpus).hexdigest()==p['input_corpus_sha256'];selection=load_corpus(corpus)
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];source=candidate_bytes(bundle,p['source_files']);assert len(source)==89
    old=snapshot_targets(ROOT/'Garnet',source,p['candidate_base']);library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';saved=ROOT/p['library']
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll'] and sha(saved)==p['library_sha256']
    cache=Path(json.loads((ROOT/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root']);assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96<<30
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    common=[prep,bundle]+[ROOT/n for n in p['archive_files']]+[ROOT/n for n in p['checkpoints']]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']];assert all(n.is_file() for n in common)
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=dict(p['files'])
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],cache_before_sha256=p['cache_before_sha256'],parents=p['parents'],phases=[],controls={},routes={},cleanup_finished=False,model_or_performance_qualified=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit source89 ordinary-output ablation inventory; protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','115 CPU controls retained unchanged. Full4K gate uses the original all48 phase-row state/vector comparisons and all128 IDs; selected repair is only its prerequisite. No graph observer outputs or timings.',common);w['cpu_exit_code']=code;save();assert code==0,'115 CPU controls failed'
        for n,b in source.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        library.write_bytes(saved.read_bytes());assert sha(library)==p['library_sha256'];inventory=p['installed_files'];assert all(sha(ROOT/n)==h for n,h in inventory.items());w['installed_source']=inventory;save()
        code=ledger.run([sys.executable,'-m','unittest',*p['mtp_test_modules']],out/'mtp-cpu-ledger','correctness','41 additional CPU controls on installed source89: original19 B1/workspace controls,8 B4 draft-study binding/ownership/geometry controls,4 MTP evidence and unchanged target-gate rejection controls. The T128 source provider is from the qualified overlay. No GPU model is executed by these symbolic tests.',common);w['mtp_cpu_exit_code']=code;save();assert code==0,'41 MTP CPU controls failed'
        phase='candidate';idle();w['phases'].append(dict(phase=phase,mode=phase,enabled=True,source_installed={n:sha(ROOT/n) for n in inventory},library_sha256=sha(library),exit_code=None));save()
        argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(phase+'-resources')),'--',sys.executable,str(M/'launch_rowwise_k_norm_mtp256_full4k.py'),str(ROOT),str(out/phase),str(cache),phase,str(ROOT/'results/b4-full4k-exact-attempt1/reference')]
        code=ledger.run(argv,out/(phase+'-ledger'),'correctness','Source89 same DLL and qualified rowwise K target with actual B4 MTP3/T16 draft prefill/T4 draft decode/full-vocabulary head. Three3968prompt+128forced-output text/image/mixed groups. Require target states/vectors/IDs vs eight B1 oracles, actual draft graphs/249prefill/127commits,16rowwise target traces and old routes. No broader MTP/prefix/quality/HTTP or speed admission.',common+[library]);w['phases'][-1]['exit_code']=code;save();idle();assert code==0,'Full4K candidate capture failed'
        log=(out/(phase+'-ledger/commands.log')).read_bytes();w['routes'][phase]={'original':route_trace(log,'off'),'rowwise':norm_trace(log,True)};save()
        candidate=json.loads((out/phase/'report.json').read_bytes());assert candidate['library_sha256']==p['library_sha256'] and candidate['rowwise_K_norm'] is candidate['primary_outputs_only'] is True
        with zipfile.ZipFile(ROOT/'results/b4-full4k-exact-attempt1-evidence.zip') as parent:
            pb='results/b4-full4k-exact-attempt1/reference/';reference=json.loads(parent.read(pb+'report.json'));assert reference['library_sha256']==p['reference_library_sha256']
            rr=lambda key,n:parent.read(pb+n);rc=lambda label,n:(out/phase/n).read_bytes()
            w['observation']=observe(selection,reference,candidate,rr,rc)
            gold={r['keys'][0]:r['generated_ids'][0] for r in reference['rows']}
            w['first_differing_ID_indices']={r['label']:[next((i for i,(a,b) in enumerate(zip(ids,gold[k])) if a!=b),None) for ids,k in zip(r['generated_ids'],r['keys'])] for r in candidate['rows']}
            (out/'analysis.json').write_bytes(json.dumps(w['observation'],indent=2).encode());save()
            w['strict_admission']=mtp_admit(selection,reference,candidate,rr,rc);save()
        w['status']='rowwise_K_norm_MTP_full4k128_exact_collected_quality_unqualified';code=0;save()
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
