"""Independent raw replay of step44/45 all-layer observations or preserved failure."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.step45_layer_boundary_analysis import analyze
from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
JOB='step45-layer-boundary-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for n,h in mf['files'].items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/step45-layer-boundary-preparation.json');p=json.loads(raw);w=json.loads(z.read(B+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored'] and type(w['exit_code']) is int
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34 and not w['model_or_performance_qualified']
        for n in p['archive_files']:assert sha(z.read(n))==p['files'][n]
        for n in ('step45_layer_boundary_analysis.py','verify_step45_layer_boundary.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][P+n]
        for mode,count in (('reference',60),('candidate',85)):
            bundle=z.read(p[mode+'_bundle']);assert sha(bundle)==p[mode+'_bundle_sha256']
            with zipfile.ZipFile(io.BytesIO(bundle)) as source:assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==count and {n:sha(source.read(n)) for n in source.namelist()}==p[mode+'_source_files']
        with zipfile.ZipFile(io.BytesIO(z.read(B+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        stages={}
        for stage in ('cpu','reference','candidate'):
            name=B+stage+'-ledger/commands.jsonl'
            if name not in mf['files']:continue
            events=[json.loads(s) for s in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p[stage+'_installed_files'];assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;stages[stage]=events[1]['exit_code']
        if stages.get('cpu')==0:assert b'Ran 66 tests' in z.read(B+'cpu-ledger/commands.log') and b'\nOK' in z.read(B+'cpu-ledger/commands.log')
        for phase in w['phases']:assert phase['source_installed']==p[phase['mode']+'_installed_files'] and phase['library_sha256']==p[phase['mode']+'_library_sha256'] and phase['exit_code']==stages[phase['mode']]
        result_analysis=None
        if 'layer_analysis' in w:
            assert assert_disabled(z.read(B+'reference-ledger/commands.log'))==w['reference_routes_absent'] and route_trace(z.read(B+'candidate-ledger/commands.log'),'off')==w['candidate_routes']
            corpus=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus)==p['input_corpus_sha256'];selection=load_corpus(corpus)
            reports={mode:json.loads(z.read(B+mode+'/report.json')) for mode in ('reference','candidate')}
            for mode,report in reports.items():assert report['library_sha256']==p[mode+'_library_sha256']
            with zipfile.ZipFile(args.proofs/'key35-coordinate-trajectory-attempt1-evidence.zip') as rp,zipfile.ZipFile(args.proofs/'key35-coordinate-trajectory-attempt1-evidence.zip') as cp:
                rb='results/key35-coordinate-trajectory-attempt1/reference/';cb='results/key35-coordinate-trajectory-attempt1/candidate/'
                result_analysis=analyze(selection,p['selected_key'],reports['reference'],reports['candidate'],json.loads(rp.read(rb+'report.json')),json.loads(cp.read(cb+'report.json')),lambda n:z.read(B+'reference/'+n),lambda n:z.read(B+'candidate/'+n),lambda n:rp.read(rb+n),lambda n:cp.read(cb+n))
            assert result_analysis==w['layer_analysis']==json.loads(z.read(B+'analysis.json'))
        passed=w['exit_code']==0
        if passed:assert w['status']=='step45_layer_boundary_collected_no_model_admission' and stages=={'cpu':0,'reference':0,'candidate':0} and result_analysis
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result={'status':'step45_layer_boundary_replayed' if passed else 'step45_layer_boundary_failure_replayed','exit_code':w['exit_code'],'error':w.get('error'),'archive_sha256':sha(archive.read_bytes()),'files_verified':len(mf['files']),'stage_exit_codes':stages,'analysis':result_analysis,'CPU_controls':66 if stages.get('cpu')==0 else None,'source_runtime_SDK_config_index_CMake_all34_baseline_restored':True,'model_or_full4K_MTP_quality_HTTP_or_performance_qualified':False,'original_full4k_still_failed':True,'two_times_target_achieved':False,'reader_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'proof_sha256':sha(raw),'layer_differences':[{k:row[k] for k in ('step','first_differing_active_layer','first_differing_full_T4_layer')} for row in result_analysis['rows']] if result_analysis else None}));return 0
if __name__=='__main__':raise SystemExit(main())
