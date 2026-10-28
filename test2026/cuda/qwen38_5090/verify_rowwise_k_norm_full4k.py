"""Replay the original full4K128 gate for source87; never promote MTP or speed."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.b4_full4k_exact_gate import observe,admit
from test2026.cuda.qwen38_5090.rowwise_k_norm_analysis import norm_trace
from test2026.cuda.qwen38_5090.b4_k_m4x4_control52_route import route_trace
JOB='rowwise-k-norm-full4k-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';B='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'))['files'];assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf)+1
        for n,h in mf.items():assert sha(z.read(n))==h
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/rowwise-k-norm-full4k-preparation.json');p=json.loads(raw);w=doc(B+'workflow.json');assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['all_restored']
        for first,last,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[first]==w[last]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34 and not w['model_or_performance_qualified']
        for n in p['archive_files']:assert sha(z.read(n))==p['files'][n]
        for n in ('verify_rowwise_k_norm_full4k.py','b4_full4k_exact_gate.py','rowwise_k_norm_analysis.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][P+n]
        assert sha(z.read(p['bundle']))==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:assert len(source.namelist())==len(set(source.namelist()))==87 and {n:sha(source.read(n)) for n in source.namelist()}==p['source_files']
        with zipfile.ZipFile(io.BytesIO(z.read(B+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(a.proofs/(job+'-verified.json')).read_bytes() and q['status']==row['status'] and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        qualified=doc('results/rowwise-k-norm-model-attempt1-verified.json');assert qualified['exit_code']==0 and qualified['analysis']['selected_step52_intervention_passed'] and set(qualified['controls'])=={'off-reference','off-candidate'}
        stages={}
        for stage in ('cpu','candidate'):
            name=B+stage+'-ledger/commands.jsonl'
            if name not in mf:continue
            events=[json.loads(s) for s in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            expected=p['files'] if stage=='cpu' else p['installed_files'];assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;stages[stage]=events[1]['exit_code']
        if stages.get('cpu')==0:assert b'Ran 115 tests' in z.read(B+'cpu-ledger/commands.log') and b'\nOK' in z.read(B+'cpu-ledger/commands.log')
        assert len(w['phases'])<=1
        for phase in w['phases']:assert phase['phase']==phase['mode']=='candidate' and phase['enabled'] is True and phase['source_installed']==p['installed_files'] and phase['library_sha256']==p['library_sha256'] and phase['exit_code']==stages['candidate']
        observation=strict=first=None;routes={}
        if 'candidate' in w['routes']:
            log=z.read(B+'candidate-ledger/commands.log');routes={'original':route_trace(log,'off'),'rowwise':norm_trace(log,True)};assert routes==w['routes']['candidate']
        if 'observation' in w:
            corpus=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus)==p['input_corpus_sha256'];selection=load_corpus(corpus)
            candidate=doc(B+'candidate/report.json');assert candidate['library_sha256']==p['library_sha256'] and candidate['rowwise_K_norm'] is candidate['primary_outputs_only'] is True
            with zipfile.ZipFile(a.proofs/'b4-full4k-exact-attempt1-evidence.zip') as parent:
                pb='results/b4-full4k-exact-attempt1/reference/';reference=json.loads(parent.read(pb+'report.json'));assert reference['library_sha256']==p['reference_library_sha256']
                rr=lambda key,n:parent.read(pb+n);rc=lambda label,n:z.read(B+'candidate/'+n)
                observation=observe(selection,reference,candidate,rr,rc);assert observation==w['observation']==doc(B+'analysis.json')
                gold={r['keys'][0]:r['generated_ids'][0] for r in reference['rows']};first={r['label']:[next((i for i,(x,y) in enumerate(zip(ids,gold[k])) if x!=y),None) for ids,k in zip(r['generated_ids'],r['keys'])] for r in candidate['rows']};assert first==w['first_differing_ID_indices']
                if 'strict_admission' in w:strict=admit(selection,reference,candidate,rr,rc);assert strict==w['strict_admission']
        passed=w['exit_code']==0
        if passed:assert w['status']=='rowwise_K_norm_full4k128_exact_collected_MTP_quality_unqualified' and stages==dict(cpu=0,candidate=0) and strict and routes
        else:assert w['status']=='failed' and isinstance(w.get('error'),str)
        result=dict(status='rowwise_K_norm_ordinary_full4k128_replayed' if passed else 'rowwise_K_norm_full4k128_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,files_verified=len(mf),stage_exit_codes=stages,observation=observation,strict_admission=strict,first_differing_ID_indices=first,routes=routes,CPU_controls=115 if stages.get('cpu')==0 else None,ordinary_full4K128_case_gate_passed=passed,source_runtime_SDK_config_index_CMake_all34_baseline_restored=True,MTP_broad_quality_HTTP_or_performance_qualified=False,original_source79_full4k_and_cache62_still_failed=True,two_times_target_achieved=False,reader_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],proof_sha256=sha(raw),ordinary_full4K128_case_gate_passed=passed)))
if __name__=='__main__':main()
