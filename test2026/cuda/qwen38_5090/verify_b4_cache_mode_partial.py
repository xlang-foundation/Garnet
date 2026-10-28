"""Replay completed full-cache process and the packed empty-state type failure only."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_b4_cache_mode_characterization import checked,A,B,JOB,METHODS
from test2026.cuda.qwen38_5090.b4_cache_mode_characterization import characterize_case,compare_capture
from test2026.cuda.qwen38_5090.bf16_batch8_external_reference import capture_exact
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);old,_=checked(args.proofs/(A+'-evidence.zip'));failed,_=checked(args.proofs/(B+'-evidence.zip'))
    with z,old,failed:
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name));w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/b4-cache-mode-characterization-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and w['exit_code']==1 and type(w['exit_code']) is int and w['cleanup_finished'] is w['library_unchanged'] is True
        assert w['last_completed_mode']=='full' and w['source_sha256']==w['source_after_sha256']==p['files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        original=json.loads((args.proofs/(JOB+'-verified.json')).read_bytes());assert original['status']=='b4_cache_mode_characterization_failure_preserved' and original['archive_sha256']==sha(archive.read_bytes())
        for job in (A,B):
            row=p['parents'][job];raw=z.read('results/'+job+'-verified.json');q=json.loads(raw)
            assert raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha(raw)==row['proof_sha256'] and q['status']==row['status']
            assert sha((args.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']==q['archive_sha256']
        for name in ('b4_cache_mode_characterization.py','bf16_batch8_external_reference.py','bf16_batch8_reference_gate.py','bf16_same_precision_gate.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][METHODS+name]
        full=doc(base+'full/report.json');packed=doc(base+'packed/report.json')
        prior_a=json.loads(old.read('results/'+A+'/model/report.json'));prior_b=json.loads(failed.read('results/'+B+'/model/report.json'))
        read_a=lambda name:old.read('results/'+A+'/model/'+name);read_b=lambda name:failed.read('results/'+B+'/model/'+name)
        read_full=lambda name:z.read(base+'full/'+name);read_packed=lambda name:z.read(base+'packed/'+name)
        assert full['status']=='b4_cache_mode_characterization_collected_needs_replay' and full['mode']=='full' and full['observed_case_count']==len(full['cases'])==1
        assert packed['status']=='failed' and packed['mode']=='packed' and packed['cases']==[] and set(packed['empty_state'])=={'0'}
        assert packed['error']=="ValueError('Unqualified same-precision state representation')"
        full_vs_b8=characterize_case(full['cases'][0],prior_a['cases'][0],[0,1,2,3],read_full,read_a)
        assert full_vs_b8==full['cases'][0]['characterization']
        full_vs_failed=characterize_case(full['cases'][0],prior_b['cases'][0],[0,1,2,3],read_full,read_b)
        for key in ('empty_state','final_reset'):
            assert set(full[key])=={'0'};capture_exact(full[key]['0'],prior_a[key],[0,1,2,3],read_full,read_a)
        for key in ('reset_state','prefill_state','final_state'):
            assert full['cases'][0][key]['logical']['reference_bf16_roundtrip_checked'] is True
        empty=packed['empty_state']['0'];assert empty['logical']['physical_words_per_row']==106823680 and empty['logical']['reference_bf16_roundtrip_checked'] is False
        try:capture_exact(empty,prior_a['empty_state'],[0,1,2,3],read_packed,read_a)
        except ValueError as error:assert str(error)=='Unqualified same-precision state representation'
        else:raise AssertionError('Original full-cache-only gate unexpectedly accepted packed representation')
        typed_empty=compare_capture(empty,prior_a['empty_state'],[0,1,2,3],read_packed,read_a);assert typed_empty['all_states_and_vectors_exact'] is True
        for mode,code in (('full',0),('packed',1)):
            resource=doc(base+mode+'-resources/resource-monitor.json');assert type(resource['child_exit_code']) is int and resource['child_exit_code']==code
            assert resource['gpu']['uuid']==p['gpu_uuid']
            events=[json.loads(line) for line in z.read(base+mode+'-gpu-ledger/commands.jsonl').splitlines()]
            assert len(events)==2 and events[-1]['event']=='finished' and events[-1]['exit_code']==code
        result={'status':'b4_cache_mode_partial_full_and_empty_state_replayed','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
                'original_failure_status':original['status'],'full_process_completed':True,'full_case_vs_b8':full_vs_b8,'full_repeat_vs_failed_full':full_vs_failed,
                'packed_empty_state':typed_empty,'packed_cases_completed':0,'original_full_only_gate_rejection_reproduced':True,
                'external_replay_verifier_sha256':sha(Path(__file__).read_bytes()),'original_strict_gate_changed':False,'model_math_or_quality_qualified':False,'cause_established':False,'performance_admitted':False,'two_times_target_achieved':False,
                'scope':'Original dual-mode owner failed and remains failed. Actual full process exited0 and its one case plus reset controls replay; packed process exited1 before any case because its empty-state capture was sent to an explicitly full-FP32-only representation gate. Separate typed compare replays the actual empty packed snapshot exactly. No packed model case or math/performance admission.'}
    target=args.proofs/(JOB+'-partial-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'full_repeat_exact':full_vs_failed['all_observed_ids_states_vectors_exact'],'full_vs_b8_exact':full_vs_b8['all_observed_ids_states_vectors_exact'],'packed_empty_typed_exact':True}));return 0

if __name__=='__main__':raise SystemExit(main())
