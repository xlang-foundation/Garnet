"""Replay raw same-DLL off/on model evidence; never promote to full4K/benchmark."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.b4_k_m4x4_control52 import control,intervention
from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
JOB='b4-k-m4x4-control52-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for n,h in mf['files'].items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/b4-k-m4x4-control52-preparation.json');p=json.loads(raw);w=json.loads(z.read(B+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored'] and type(w['exit_code']) is int
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34
        assert w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted'] and not w['full128_output_4K_MTP_quality_qualified']
        for n in p['archive_files']:assert sha(z.read(n))==p['files'][n]
        for n in ('b4_k_m4x4_control52.py','b4_k_m4x4_control52_route.py','verify_b4_k_m4x4_control52.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][P+n]
        assert sha(z.read(p['bundle']))==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:assert source.testzip() is None and len(source.namelist())==85 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(B+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        build=json.loads(z.read('results/b4-k-m4x4-build-attempt1-verified.json'));assert build['exit_code']==0 and build['candidate_library_sha256']==p['library_sha256'] and build['candidate_files']==85
        corpus=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus)==p['input_corpus_sha256'];selection=load_corpus(corpus)
        stages={}
        for stage in ('cpu','off','on'):
            n=B+stage+'-ledger/commands.jsonl'
            if n not in mf['files']:continue
            events=[json.loads(line) for line in z.read(n).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p['installed_files'];assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;stages[stage]=events[1]['exit_code']
        if 'cpu' in stages:assert stages['cpu']==w['cpu_exit_code']
        if stages.get('cpu')==0:assert b'Ran 47 tests' in z.read(B+'cpu-ledger/commands.log') and b'\nOK' in z.read(B+'cpu-ledger/commands.log')
        for phase in w['phases']:
            policy=phase['policy'];assert policy in ('off','on') and phase['source_installed']==p['installed_files'] and phase['library_sha256']==p['library_sha256'] and phase['flags']=={**p['flags'],'GARNET_FP8_K_BATCH4_DECODE_M4X4':'1' if policy=='on' else '0'} and phase['exit_code']==stages[policy]
        off_result=on_result=None
        with zipfile.ZipFile(args.proofs/'b4-k35-cache52-attempt1-evidence.zip') as parent:
            pb='results/b4-k35-cache52-attempt1/';reference=json.loads(parent.read(pb+'reference/report.json'));old=json.loads(parent.read(pb+'candidate/report.json'))
            if 'off_control' in w:
                report=json.loads(z.read(B+'off/report.json'));assert report['library_sha256']==p['library_sha256'];off_result=control(report,old,lambda n:z.read(B+'off/'+n),lambda n:parent.read(pb+'candidate/'+n),p['library_sha256'],p['original_candidate_library_sha256']);assert off_result==w['off_control']==json.loads(z.read(B+'off-control.json'))
            for policy in ('off','on'):
                if policy+'_route' in w:assert route_trace(z.read(B+policy+'-ledger/commands.log'),policy)==w[policy+'_route']
            if 'intervention' in w:
                report=json.loads(z.read(B+'on/report.json'));assert report['library_sha256']==p['library_sha256']
                with zipfile.ZipFile(args.proofs/'b4-full4k-exact-attempt1-evidence.zip') as full,zipfile.ZipFile(args.proofs/'b4-full4k-decode-prefix-attempt1-evidence.zip') as prefix:
                    fb='results/b4-full4k-exact-attempt1/reference/';eb='results/b4-full4k-decode-prefix-attempt1/candidate/'
                    on_result=intervention(selection,p['selected_key'],reference,report,json.loads(full.read(fb+'report.json')),json.loads(prefix.read(eb+'report.json')),lambda n:parent.read(pb+'reference/'+n),lambda n:z.read(B+'on/'+n),lambda n:full.read(fb+n),lambda n:prefix.read(eb+n))
                assert on_result==w['intervention']==json.loads(z.read(B+'intervention.json'))
        passed=w['exit_code']==0
        if passed:assert w['status']=='same_DLL_control52_selected_intervention_passed_full4K_unqualified' and stages=={'cpu':0,'off':0,'on':0} and off_result and on_result['selected_step52_intervention_passed'] and all(k+'_route' in w for k in ('off','on'))
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result={'status':'same_DLL_control52_selected_intervention_replayed' if passed else 'same_DLL_control52_failure_replayed','exit_code':w['exit_code'],'error':w.get('error'),'archive_sha256':sha(archive.read_bytes()),'files_verified':len(mf['files']),'stage_exit_codes':stages,'off_control':off_result,'intervention':on_result,'library_sha256':p['library_sha256'],'CPU_controls':47 if stages.get('cpu')==0 else None,'source_runtime_SDK_config_index_CMake_all34_baseline_restored':True,'immutable_reference_reused_no_fresh_B1':True,'full128_output_4K_MTP_quality_HTTP_or_performance_qualified':False,'original_full4k_still_failed':True,'two_times_target_achieved':False,'reader_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'proof_sha256':sha(raw),'selected_intervention_passed':bool(on_result and on_result['selected_step52_intervention_passed'])}));return 0
if __name__=='__main__':raise SystemExit(main())
