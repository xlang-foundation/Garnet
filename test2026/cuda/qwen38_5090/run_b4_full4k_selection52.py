"""Exclusive full4K text/image/mixed B4-vs-eight-B1 admission; no timing."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='b4-full4k-selection52-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import snapshot_targets,restore_targets,candidate_bytes
    prep=M/'evidence/b4-full4k-selection52-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='B4_full4k_selection52_prepared_GPU_unrun'
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
    corpus=(M/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert hashlib.sha256(corpus).hexdigest()==p['input_corpus_sha256'];selection=load_corpus(corpus)
    for n,h in {**p['files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert type(w['exit_code']) is int and w['exit_code']==row['exit_code'] and w['exit_code'] in (0,1) and w['cleanup_finished']
    for n,h in p['cuda_objects'].items():assert sha(ROOT/n)==h
    assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256']
    bundles={mode:ROOT/p[mode+'_bundle'] for mode in ('reference','candidate')}
    for mode,bundle in bundles.items():assert sha(bundle)==p[mode+'_bundle_sha256']
    sources={mode:candidate_bytes(bundle,p[mode+'_source_files']) for mode,bundle in bundles.items()};assert len(sources['reference'])==60 and len(sources['candidate'])==79
    old=snapshot_targets(ROOT/'Garnet',sources['candidate'],p['candidate_base']);baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';library=ROOT/'build/bin/garnet.dll'
    assert sha(baseline)==sha(library)==p['runtime']['garnet.dll']
    saved={mode:ROOT/p[mode+'_library'] for mode in sources}
    for mode,path in saved.items():assert sha(path)==p[mode+'_library_sha256']
    receipt=ROOT/'results/study-ssd-engine-cache/receipt.json';cache=Path(json.loads(receipt.read_bytes())['cache_root']);assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96<<30
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();assert shutil.disk_usage(ROOT).free>=4<<30
    out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=dict(p['files'])
    common=[prep,*bundles.values()]+[ROOT/n for n in p['archive_files']]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']]
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],candidate_base=p['candidate_base'],parents=p['parents'],cleanup_finished=False,performance_admitted=False,full4k_MTP_805_qualified=False,phases=[],cuda_objects_before=p['cuda_objects'],cache_before_sha256=p['cache_before_sha256'])
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit B4-vs-four-B1 short mixed numerical admission; protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_b4_b1_short_m16_gate','test2026.cuda.qwen38_5090.test_b4_v_m16_short_route','test2026.cuda.qwen38_5090.test_b4_qk_m16_short_route','test2026.cuda.qwen38_5090.test_b4_full4k_inputs','test2026.cuda.qwen38_5090.test_b4_full4k_exact_gate','test2026.cuda.qwen38_5090.test_b4_full4k_decode_prefix','test2026.cuda.qwen38_5090.test_b4_full4k_selection52'],out/'cpu-ledger','correctness','Strict complete128component/finite raw vector/signedzero/rowpermutation/slot/ID corruption gates',common);assert code==0,'CPU gate controls failed'
        for mode in ('reference','candidate'):
            idle()
            for n,b in sources[mode].items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
            inventory={**p['files'],**{'Garnet/'+n:h for n,h in p[mode+'_source_files'].items()}};assert inventory==p[mode+'_installed_files']
            library.write_bytes(saved[mode].read_bytes());assert sha(library)==p[mode+'_library_sha256'];w['phases'].append(dict(mode=mode,source_installed={n:sha(ROOT/n) for n in inventory},library_sha256=sha(library),exit_code=None));save()
            argv=[sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(mode+'-resources')),'--',sys.executable,str(M/'launch_b4_full4k_selection52.py'),str(ROOT),str(out/mode),str(cache),mode,str(out/'reference')]
            code=ledger.run(argv,out/(mode+'-ledger'),'correctness','One B1 reference for firstmixedbatch differingID52 row, thenwholeB4mixedgroup. Originalmodelgraphs/weights/precision/head/nativeDLLs unchanged,3968prompt+53outputs/cap4096. Complete128component/physical-logicalhash/finitehidden-normalizedvectors after7and52steps. All52consumedIDs mustmatch; ID52 isunconsumed. Match53same-profileIDs+endpoints againstoriginalfull4Kfailedparent; matchallcurrentstep7primaryrowsagainstqualifiedearlyprefix.33CPU/actualGDN-V-QKroute/restoration guards; no observer/kernel/nativebuild/timing/quality/performanceadmission.',common+[library]);w['phases'][-1]['exit_code']=code;save();idle()
            if code:raise RuntimeError(mode+' numerical admission failed; preserve raw witnesses and errors')
        from test2026.cuda.qwen38_5090.b4_b1_short_m16_gate import route_trace
        from test2026.cuda.qwen38_5090.b4_full4k_selection52 import observe,same_profile_full_parent,same_profile_step7,first_mixed_difference
        reference=json.loads((out/'reference/report.json').read_bytes());candidate=json.loads((out/'candidate/report.json').read_bytes())
        read_reference=lambda key,name:(out/'reference'/name).read_bytes();read_candidate=lambda name:(out/'candidate'/name).read_bytes()
        with zipfile.ZipFile(ROOT/'results/b4-full4k-exact-attempt1-evidence.zip') as parent:
            base='results/b4-full4k-exact-attempt1/'
            old_reference=json.loads(parent.read(base+'reference/report.json'));old_candidate=json.loads(parent.read(base+'candidate/report.json'))
            assert first_mixed_difference(selection,old_reference,old_candidate)==p['selection_from_original']
            w['same_profile_parent']={mode:same_profile_full_parent(report,old_report,lambda name,mode=mode:(out/mode/name).read_bytes(),lambda name,mode=mode:parent.read(base+mode+'/'+name)) for mode,report,old_report in (('reference',reference,old_reference),('candidate',candidate,old_candidate))}
        (out/'same-profile-parent.json').write_bytes(json.dumps(w['same_profile_parent'],indent=2).encode());save()
        with zipfile.ZipFile(ROOT/'results/b4-full4k-decode-prefix-attempt1-evidence.zip') as parent:
            base='results/b4-full4k-decode-prefix-attempt1/'
            w['step7_parent_parity']={mode:same_profile_step7(report,json.loads(parent.read(base+mode+'/report.json')),lambda name,mode=mode:(out/mode/name).read_bytes(),lambda name,mode=mode:parent.read(base+mode+'/'+name)) for mode,report in (('reference',reference),('candidate',candidate))}
        (out/'step7-parent-parity.json').write_bytes(json.dumps(w['step7_parent_parity'],indent=2).encode());save()
        w['observation']=observe(selection,p['selected_key'],reference,candidate,read_reference,read_candidate);(out/'analysis.json').write_bytes(json.dumps(w['observation'],indent=2).encode());save()
        w['actual_GDN_M16_route']=route_trace((out/'candidate-ledger/commands.log').read_bytes());(out/'route-analysis.json').write_bytes(json.dumps(w['actual_GDN_M16_route'],indent=2).encode());save()
        from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import route_trace as full_route
        from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
        w['reference_no_V_or_QK_route']=assert_disabled((out/'reference-ledger/commands.log').read_bytes());w['actual_GDN_V_QK_M16_route']=full_route((out/'candidate-ledger/commands.log').read_bytes());(out/'full-route-analysis.json').write_bytes(json.dumps(w['actual_GDN_V_QK_M16_route'],indent=2).encode());save()
        w['original_full4k_admission_still_failed']=True;save()
        w['status']='selected_step52_diagnostic_collected_no_numerical_admission';code=0
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();library.write_bytes(baseline.read_bytes())
        try:
            if installed:restore_targets(ROOT/'Garnet',installed,old)
            w['baseline_source_restored']=True
        except BaseException as error:w.update(baseline_source_restored=False,restoration_error=repr(error));code=1
        w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in p['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in p['runtime']},environment_after={n:sha(Path(n)) for n in p['environment']},checkpoint_after={n:sha(ROOT/n) for n in p['checkpoints']},baseline_library_restored=sha(library)==p['runtime']['garnet.dll']);w['retired_private_engine_files']=retire(cache,ROOT,before,started);w['cleanup_finished']=True;w['cuda_objects_after']={n:sha(ROOT/n) for n in p['cuda_objects']};w['cache_after_sha256']=sha(ROOT/'build/CMakeCache.txt');w['build_artifacts_unchanged']=w['cuda_objects_before']==w['cuda_objects_after'] and w['cache_before_sha256']==w['cache_after_sha256'];save()
        if not w['build_artifacts_unchanged']:code=1;w.update(status='failed',exit_code=1,error='Production build artifacts changed during model gate');save()
        files={n.relative_to(ROOT).as_posix():n for n in [*[n for n in out.rglob('*') if n.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in files.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in files.items()},scope=p['scope']),indent=2))
        print(json.dumps(dict(status=w['status'],exit_code=code,archive_sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
