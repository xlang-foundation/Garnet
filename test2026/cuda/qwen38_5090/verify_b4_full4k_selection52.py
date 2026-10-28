"""Independent full4K8B1/3B4 complete-state/vector/128ID and M16 route replay."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_b1_short_m16_gate import route_trace
from test2026.cuda.qwen38_5090.b4_full4k_selection52 import observe,same_profile_full_parent,same_profile_step7,first_mixed_difference,STATUS
from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
from test2026.cuda.qwen38_5090.fa_value_sdk_control import verify_derivations
JOB='b4-full4k-selection52-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';BASE='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for n,h in mf['files'].items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/b4-full4k-selection52-preparation.json');p=json.loads(raw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['cuda_objects_before']==w['cuda_objects_after']==p['cuda_objects'] and w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and w['build_artifacts_unchanged']
        assert w['performance_admitted'] is w['full4k_MTP_805_qualified'] is False
        for n,h in p['archive_files'].items():assert sha(z.read(n))==h
        for n in ('b4_b1_short_m16_gate.py','b4_b1_exact_gate.py','verify_b4_full4k_exact.py','b4_full4k_exact_gate.py','b4_full4k_inputs.py','b4_full4k_decode_prefix.py','b4_full4k_selection52.py','verify_b4_full4k_selection52.py'):assert Path(__file__).with_name(n).read_bytes()==z.read(P+n)
        for mode in ('reference','candidate'):
            assert sha(z.read(p[mode+'_bundle']))==p[mode+'_bundle_sha256']
            with zipfile.ZipFile(io.BytesIO(z.read(p[mode+'_bundle']))) as source:assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==len(p[mode+'_source_files']) and {n:sha(source.read(n)) for n in source.namelist()}==p[mode+'_source_files']
        for job,row in p['parents'].items():
            raw=z.read('results/'+job+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and raw==(a.proofs/(job+'-verified.json')).read_bytes() and sha((a.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']
        verify_derivations(json.loads(z.read(P+'evidence/b4-qk-m16-short-owner-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/b4-qk-m16-short-v2-owner-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/b4-full4k-exact-owner-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/b4-full4k-exact-probe-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/b4-full4k-decode-prefix-derivation.json')),lambda n:z.read(P+n))
        verify_derivations(json.loads(z.read(P+'evidence/b4-full4k-selection52-derivation.json')),lambda n:z.read(P+n))
        corpus=z.read(P+'evidence/fixed-length-4096-corpus.zip');assert sha(corpus)==p['input_corpus_sha256'];selection=load_corpus(corpus);inputs=selection['inputs'];assert len(inputs)==8 and all(len(r['prompt_ids'])==3968 for r in inputs.values())
        def commands(name,expected):
            events=[json.loads(line) for line in z.read(BASE+name+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int;assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected;return events[1]['exit_code']
        cpu_code=commands('cpu',p['files'])
        if cpu_code==0:assert b'Ran 33 tests' in z.read(BASE+'cpu-ledger/commands.log') and b'\nOK' in z.read(BASE+'cpu-ledger/commands.log')
        else:assert w['exit_code']!=0 and not w['phases'] and 'CPU gate controls failed' in w['error']
        reports={}
        for phase in w['phases']:
            mode=phase['mode'];assert mode not in reports and mode in ('reference','candidate') and phase['source_installed']==p[mode+'_installed_files'] and phase['library_sha256']==p[mode+'_library_sha256'] and commands(mode,p[mode+'_installed_files'])==phase['exit_code']
            name=BASE+mode+'/report.json'
            if name in mf['files']:
                report=json.loads(z.read(name));assert report['inputs']==inputs and report['library_sha256']==p[mode+'_library_sha256'];reports[mode]=report
        observation=route=full=None;route_error=None;strict=False
        if set(reports)=={'reference','candidate'} and reports['reference']['status']==reports['candidate']['status']==STATUS:
            reference=reports['reference'];candidate=reports['candidate'];r=lambda key,n:z.read(BASE+'reference/'+n);c=lambda n:z.read(BASE+'candidate/'+n)
            with zipfile.ZipFile(a.proofs/'b4-full4k-exact-attempt1-evidence.zip') as parent:
                base='results/b4-full4k-exact-attempt1/'
                old_reference=json.loads(parent.read(base+'reference/report.json'));old_candidate=json.loads(parent.read(base+'candidate/report.json'))
                assert first_mixed_difference(selection,old_reference,old_candidate)==p['selection_from_original']
                parity={mode:same_profile_full_parent(report,old_report,lambda n,mode=mode:z.read(BASE+mode+'/'+n),lambda n,mode=mode:parent.read(base+mode+'/'+n)) for mode,report,old_report in (('reference',reference,old_reference),('candidate',candidate,old_candidate))}
            assert parity==w['same_profile_parent']==json.loads(z.read(BASE+'same-profile-parent.json'))
            with zipfile.ZipFile(a.proofs/'b4-full4k-decode-prefix-attempt1-evidence.zip') as parent:
                base='results/b4-full4k-decode-prefix-attempt1/'
                early={mode:same_profile_step7(report,json.loads(parent.read(base+mode+'/report.json')),lambda n,mode=mode:z.read(BASE+mode+'/'+n),lambda n,mode=mode:parent.read(base+mode+'/'+n)) for mode,report in (('reference',reference),('candidate',candidate))}
            assert early==w['step7_parent_parity']==json.loads(z.read(BASE+'step7-parent-parity.json'))
            observation=observe(selection,p['selected_key'],reference,candidate,r,c);assert observation==w['observation']==json.loads(z.read(BASE+'analysis.json'))
            assert b'GARNET_GDNM16_FACTORY ' not in z.read(BASE+'reference-ledger/commands.log')
            try:route=route_trace(z.read(BASE+'candidate-ledger/commands.log'))
            except AssertionError as error:route_error=repr(error);assert w['exit_code']!=0 and 'actual_GDN_M16_route' not in w
            if route is not None:assert route==w['actual_GDN_M16_route']==json.loads(z.read(BASE+'route-analysis.json'))
            from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import route_trace as full_route
            from test2026.cuda.qwen38_5090.b4_qk_m16_short_route import assert_disabled
            try:
                assert assert_disabled(z.read(BASE+'reference-ledger/commands.log'))==w['reference_no_V_or_QK_route'];full=full_route(z.read(BASE+'candidate-ledger/commands.log'))
            except AssertionError as error:route_error=repr(error);assert w['exit_code']!=0 and 'actual_GDN_V_QK_M16_route' not in w
            if full is not None:assert full==w['actual_GDN_V_QK_M16_route']==json.loads(z.read(BASE+'full-route-analysis.json'))
            if w['exit_code']==0:assert full is not None and w['original_full4k_admission_still_failed'];strict=True
        if w['exit_code']==0:assert w['status']=='selected_step52_diagnostic_collected_no_numerical_admission' and [v['mode'] for v in w['phases']]==['reference','candidate'] and all(v['exit_code']==0 for v in w['phases']) and strict and route is not None
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        result=dict(status='selected_step52_diagnostic_replayed' if strict else 'selected_step52_job_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),candidate_library_sha256=p['candidate_library_sha256'],reference_library_sha256=p['reference_library_sha256'],phase_statuses={k:v['status'] for k,v in reports.items()},phase_errors={k:v.get('error') for k,v in reports.items()},observation=observation,actual_GDN_M16_route=route,actual_GDN_V_QK_M16_route=full,route_error=route_error,strict_full4k_admission_passed=False,diagnostic_completed=strict,source_runtime_SDK_checkpoint_CMake_all34objects_restored=True,original_failed_job_promoted=False,full4k_exact_gate_only=True,full4k_state_IDs_qualified=False,MTP_805_HTTP_or_performance_qualified=False,two_times_target_achieved=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],proof_sha256=sha(raw),strict_full4k_admission_passed=False,diagnostic_completed=strict)));return 0
if __name__=='__main__':raise SystemExit(main())
