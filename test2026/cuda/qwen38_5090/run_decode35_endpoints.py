"""Two fresh six-endpoint on processes; exact source/runtime-pinned ordinary off parents reused."""
from pathlib import Path
import hashlib,json,os,shutil,sys,time,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='decode35-endpoints-attempt1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from run_bf16_batch8_admission import retire
    from run_z128_factory_trace_native_build import candidate_bytes,snapshot_targets,restore_targets
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
    from test2026.cuda.qwen38_5090.decode35_endpoint_analysis import PHASES,primary_control,witness_trace,analyze
    from test2026.cuda.qwen38_5090.decode35_endpoint_reuse import checked_reuse
    from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
    from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
    prep=M/'evidence/decode35-endpoints-preparation.json';p=json.loads(prep.read_bytes());assert p['status']=='decode35_endpoints_prepared_GPU_unrun'
    for n,h in {**p['files'],**p['checkpoints'],**p['cuda_objects']}.items():assert sha(ROOT/n)==h,('Preflight drift',n)
    for n,h in p['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in p['environment'].items():assert sha(Path(n))==h
    assert sha(ROOT/'build/CMakeCache.txt')==p['cache_before_sha256']
    for job,row in p['parents'].items():
        path=ROOT/'results'/(job+'-verified.json');q=json.loads(path.read_bytes());assert sha(path)==row['proof_sha256'] and q['status']==row['status'] and sha(ROOT/'results'/(job+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes());assert w['exit_code']==row['exit_code'] and w['cleanup_finished']
    corpus=(M/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert hashlib.sha256(corpus).hexdigest()==p['input_corpus_sha256'];selection=load_corpus(corpus)
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256'];source=candidate_bytes(bundle,p['source_files']);assert len(source)==86
    old=snapshot_targets(ROOT/'Garnet',source,p['candidate_base']);library=ROOT/'build/bin/garnet.dll';baseline=ROOT/'build/bin/garnet-native-tokenizer-nfc-attempt1.dll';saved=ROOT/p['library']
    assert sha(library)==sha(baseline)==p['runtime']['garnet.dll'] and sha(saved)==p['library_sha256']
    cache=Path(json.loads((ROOT/'results/study-ssd-engine-cache/receipt.json').read_bytes())['cache_root']);assert cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(ROOT) and shutil.disk_usage(cache).free>=96<<30
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists()
    common=[prep,bundle]+[ROOT/n for n in p['archive_files']]+[ROOT/n for n in p['checkpoints']]+[ROOT/'results'/(n+'-verified.json') for n in p['parents']];assert all(n.is_file() for n in common),'Missing evidence dependency'
    idle();out.mkdir();os.chdir(ROOT/'Garnet');before={n.name for n in cache.iterdir()};started=time.time();code=1;installed={};inventory=dict(p['files'])
    w=dict(status='running',preparation_sha256=sha(prep),source_before=p['files'],runtime_before=p['runtime'],environment_before=p['environment'],checkpoint_before=p['checkpoints'],cuda_objects_before=p['cuda_objects'],cache_before_sha256=p['cache_before_sha256'],parents=p['parents'],phases=[],controls={},routes={},cleanup_finished=False,model_or_performance_qualified=False)
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit same-DLL operand model source inventory; protected unrelated bodies excluded','owned_source_sha256':{n:sha(ROOT/n) for n in inventory}}
    with zipfile.ZipFile(out/'original-source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for n,b in old.items():z.writestr(n,b)
    save()
    try:
        code=ledger.run([sys.executable,'-m','unittest',*p['test_modules']],out/'cpu-ledger','correctness','97CPU controls: all81 seven-stage controls unchanged,6 endpoint topology,7 endpoint analysis,3 immutable off reuse. Native hook remains7; no pre-RoPE suffix claim for six endpoints. Full all-row primary/cache/timeline/layer gates retained.',common);w['cpu_exit_code']=code;save();assert code==0,'97 CPU controls failed'
        for n,b in source.items():installed[n]=b;(ROOT/'Garnet'/n).write_bytes(b)
        library.write_bytes(saved.read_bytes());assert sha(library)==p['library_sha256'];inventory=p['installed_files'];assert all(sha(ROOT/n)==h for n,h in inventory.items())
        reports={};reads={}
        with zipfile.ZipFile(ROOT/'results/step45-layer-boundary-attempt2-evidence.zip') as parent,zipfile.ZipFile(ROOT/'results/decode35-operands-attempt1-evidence.zip') as ordinary:
            parent_base='results/step45-layer-boundary-attempt2/'
            historical={role:json.loads(parent.read(parent_base+role+'/report.json')) for role in ('reference','candidate')}
            historical_reads={role:(lambda n,role=role:parent.read(parent_base+role+'/'+n)) for role in historical}
            offproof=json.loads((ROOT/'results/decode35-operands-attempt1-verified.json').read_bytes())
            reports,reads,w['reused_off']=checked_reuse(p,ordinary,offproof)
            for phase in PHASES[:2]:
                role=phase.split('-')[1]
                w['controls'][phase]=primary_control(reports[phase],historical[role],reads[phase],historical_reads[role],p['selected_key'],p['library_sha256'],p['historical_libraries'][role],False,True)
                assert w['controls'][phase]==offproof['controls'][phase]
            save()
            for phase in PHASES[2:]:
                idle();role=phase.split('-')[1];enabled=phase.startswith('on-');batch=1 if role=='reference' else 4
                w['phases'].append(dict(phase=phase,mode=role,enabled=enabled,source_installed={n:sha(ROOT/n) for n in inventory},library_sha256=sha(library),exit_code=None));save()
                code=ledger.run([sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(phase+'-resources')),'--',sys.executable,str(M/'launch_decode35_endpoints.py'),str(ROOT),str(out/phase),str(cache),phase,str(out/'off-reference')],out/(phase+'-ledger'),'correctness','Same source86DLL and seven-stage native hook; six final endpoints omit pre-RoPE. Fresh on process must reproduce immutable same-DLL ordinary off: all53IDs/five all-row states and vectors/full8MiBcache/timeline/all64 layers44-45 before interpreting six endpoints. Off parents separately rechecked against historicalsource60/85. No performance admission.',common+[library]);w['phases'][-1]['exit_code']=code;save();idle();assert code==0,phase+' capture failed'
                log=(out/(phase+'-ledger')/'commands.log').read_bytes();w['routes'][phase]={'original':assert_disabled(log) if role=='reference' else route_trace(log,'off'),'decode35':witness_trace(log,batch,enabled)}
                reports[phase]=json.loads((out/phase/'report.json').read_bytes());reads[phase]=lambda n,phase=phase:(out/phase/n).read_bytes()
                if enabled:w['controls'][phase]=primary_control(reports[phase],reports['off-'+role],reads[phase],reads['off-'+role],p['selected_key'],p['library_sha256'],p['library_sha256'],True,False)
                else:w['controls'][phase]=primary_control(reports[phase],historical[role],reads[phase],historical_reads[role],p['selected_key'],p['library_sha256'],p['historical_libraries'][role],False,True)
                save()
            w['operand_analysis']=analyze(selection,p['selected_key'],reports,historical,reads,historical_reads,p['library_sha256'],p['historical_libraries'])
        (out/'analysis.json').write_bytes(json.dumps(w['operand_analysis'],indent=2).encode());w['status']='same_DLL_decode35_endpoints_collected_unqualified';code=0;save()
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
