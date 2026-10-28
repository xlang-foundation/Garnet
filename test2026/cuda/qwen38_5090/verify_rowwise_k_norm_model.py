"""Replay source87 ordinary-output model evidence, retaining any numerical failure."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.rowwise_k_norm_analysis import PHASES,control,norm_trace,analyze
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
JOB='rowwise-k-norm-model-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/rowwise-k-norm-model-preparation.json');p=json.loads(raw);w=doc(B+'workflow.json');assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored']
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34 and not w['model_or_performance_qualified']
        for n in p['archive_files']:assert sha(z.read(n))==p['files'][n]
        for n in ('rowwise_k_norm_analysis.py','verify_rowwise_k_norm_model.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][P+n]
        assert sha(z.read(p['bundle']))==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:assert len(source.namelist())==len(set(source.namelist()))==87 and {n:sha(source.read(n)) for n in source.namelist()}==p['source_files']
        with zipfile.ZipFile(io.BytesIO(z.read(B+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(a.proofs/(job+'-verified.json')).read_bytes() and q['status']==row['status'] and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        stages={}
        for stage in ('cpu',*PHASES):
            name=B+stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            events=[json.loads(s) for s in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p['installed_files'];assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;stages[stage]=events[1]['exit_code']
        if stages.get('cpu')==0:assert b'Ran 115 tests' in z.read(B+'cpu-ledger/commands.log') and b'\nOK' in z.read(B+'cpu-ledger/commands.log')
        assert [r['phase'] for r in w['phases']]==list(PHASES[:len(w['phases'])])
        for phase in w['phases']:assert phase['source_installed']==p['installed_files'] and phase['library_sha256']==p['library_sha256'] and phase['exit_code']==stages[phase['phase']]
        reports={};reads={};controls={};routes={};result_analysis=None
        with zipfile.ZipFile(a.proofs/'key35-coordinate-trajectory-attempt1-evidence.zip') as parent:
            pb='results/key35-coordinate-trajectory-attempt1/'
            for phase in PHASES:
                if stages.get(phase)!=0:continue
                role=phase.split('-')[1];enabled=phase.startswith('on-');reports[phase]=doc(B+phase+'/report.json');reads[phase]=lambda n,phase=phase:z.read(B+phase+'/'+n)
                if phase in w['routes']:
                    log=z.read(B+phase+'-ledger/commands.log');routes[phase]={'original':assert_disabled(log) if role=='reference' else route_trace(log,'off'),'rowwise':norm_trace(log,enabled)};assert routes[phase]==w['routes'][phase]
                if phase in w['controls']:
                    old=json.loads(parent.read(pb+role+'/report.json'));controls[phase]=control(reports[phase],old,reads[phase],lambda n,role=role:parent.read(pb+role+'/'+n),p['library_sha256'],p['historical_libraries'][role]);assert controls[phase]==w['controls'][phase]
        if 'on-candidate' in stages:assert set(controls)==set(PHASES[:2])
        if 'analysis' in w:
            assert set(reports)==set(routes)==set(PHASES)
            corpus=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus)==p['input_corpus_sha256']
            with zipfile.ZipFile(a.proofs/'b4-full4k-exact-attempt1-evidence.zip') as full,zipfile.ZipFile(a.proofs/'b4-full4k-decode-prefix-attempt1-evidence.zip') as prefix:
                fb='results/b4-full4k-exact-attempt1/reference/';bb='results/b4-full4k-decode-prefix-attempt1/candidate/'
                result_analysis=analyze(load_corpus(corpus),p['selected_key'],reports,reads,json.loads(full.read(fb+'report.json')),json.loads(prefix.read(bb+'report.json')),lambda n:full.read(fb+n),lambda n:prefix.read(bb+n),p['library_sha256'])
            assert result_analysis==w['analysis']==doc(B+'analysis.json')
        passed=w['exit_code']==0
        if passed:assert w['status']=='rowwise_K_norm_selected_intervention_passed_full4K_unqualified' and stages==dict(cpu=0,**{n:0 for n in PHASES}) and result_analysis['selected_step52_intervention_passed']
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result=dict(status='rowwise_K_norm_selected_model_replayed' if passed else 'rowwise_K_norm_model_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,files_verified=len(mf),stage_exit_codes=stages,controls=controls,routes=routes,analysis=result_analysis,CPU_controls=115 if stages.get('cpu')==0 else None,source_runtime_SDK_config_index_CMake_all34_baseline_restored=True,model_or_full4K_MTP_quality_HTTP_or_performance_qualified=False,original_full4k_still_failed=True,two_times_target_achieved=False,reader_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps({'status':result['status'],'exit_code':result['exit_code'],'proof_sha256':sha(raw),'selected_intervention_passed':result_analysis['selected_step52_intervention_passed'] if result_analysis else None}))
if __name__=='__main__':main()
