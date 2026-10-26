"""Replay packed-only case against the actual preserved full-cache companion and B8."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_b4_cache_mode_characterization import checked,A,B,METHODS
from test2026.cuda.qwen38_5090.b4_cache_mode_characterization import characterize_case,compare_capture
JOB='b4-packed-cache-characterization-attempt1';C='b4-cache-mode-characterization-attempt1'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive)
    previous={};observations={};memory=None
    try:
        for job in (A,B,C):previous[job]=checked(args.proofs/(job+'-evidence.zip'))[0]
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name));w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/b4-packed-cache-characterization-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['library_unchanged'] is True
        assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime']
        assert w['environment_before']==w['environment_after']==p['environment'] and w['parents']==p['parents']
        for job,old in previous.items():
            row=p['parents'][job];raw=z.read('results/'+job+'-verified.json');q=json.loads(raw)
            assert raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha(raw)==row['proof_sha256'] and q['status']==row['status']
            assert sha((args.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            cw=json.loads(old.read('results/'+job+'/workflow.json'));assert type(cw['exit_code']) is int and cw['exit_code']==row['exit_code'] and cw['cleanup_finished'] is True
        old=previous[C];cw=json.loads(old.read('results/'+C+'/workflow.json'));cprep=old.read(METHODS+'evidence/b4-cache-mode-characterization-preparation.json')
        partial_raw=z.read('results/'+C+'-partial-verified.json');partial=json.loads(partial_raw)
        assert partial_raw==(args.proofs/(C+'-partial-verified.json')).read_bytes() and sha(partial_raw)==p['partial_parent']['proof_sha256']
        assert partial['status']=='b4_cache_mode_partial_full_and_empty_state_replayed' and partial['archive_sha256']==p['parents'][C]['archive_sha256']
        assert partial['external_replay_verifier_sha256']==p['new_files'][METHODS+'verify_b4_cache_mode_partial.py']
        assert p['files']=={**cw['source_after_sha256'],METHODS+'evidence/b4-cache-mode-characterization-preparation.json':sha(cprep),'results/'+C+'-partial-verified.json':sha(partial_raw),**p['new_files']}
        assert cw['runtime_dependencies']==cw['runtime_dependencies_after']==p['runtime'] and cw['environment_before']==cw['environment_after']==p['environment']
        for name,digest in p['new_files'].items():assert sha(z.read(name))==digest
        for name in ('b4_cache_mode_characterization.py','verify_b4_packed_cache_characterization.py','verify_b4_cache_mode_partial.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][METHODS+name]
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='b4_packed_cache_characterization_collected_needs_replay' and w['last_completed_mode']=='packed'
            report=doc(base+'packed/report.json');full=json.loads(old.read('results/'+C+'/full/report.json'))
            assert report['status']=='b4_packed_cache_characterization_collected_needs_replay' and report['mode']=='packed'
            assert (report['physical_batch'],report['capacity'],report['prefill_chunk'],report['decode_chunk'])==(4,4096,32,4)
            assert report['bf16_kv_float_reference'] is False and report['compact_commit_used'] is True
            assert report['mtp_executed'] is report['http_executed'] is report['performance_admitted'] is False
            assert report['library_sha256']==full['library_sha256']==p['runtime']['garnet.dll']
            assert report['observed_case_count']==len(report['cases'])==1 and len(full['cases'])==1
            case=report['cases'][0];assert case['physical_rows']==[0,1,2,3] and case['baseline_case_index']==case['half']==0
            prior_a=json.loads(previous[A].read('results/'+A+'/model/report.json'));prior_b=json.loads(previous[B].read('results/'+B+'/model/report.json'))
            read=lambda name:z.read(base+'packed/'+name);read_a=lambda name:previous[A].read('results/'+A+'/model/'+name)
            read_b=lambda name:previous[B].read('results/'+B+'/model/'+name);read_full=lambda name:old.read('results/'+C+'/full/'+name)
            observations['packed_vs_b8']=characterize_case(case,prior_a['cases'][0],[0,1,2,3],read,read_a)
            assert observations['packed_vs_b8']==case['characterization'] and case['all_reference_rows_exact'] is observations['packed_vs_b8']['all_observed_ids_states_vectors_exact']
            observations['packed_vs_full_repeat']=characterize_case(case,full['cases'][0],[0,1,2,3],read,read_full)
            observations['full_repeat_vs_failed_full']=characterize_case(full['cases'][0],prior_b['cases'][0],[0,1,2,3],read_full,read_b)
            assert observations['full_repeat_vs_failed_full']==partial['full_repeat_vs_failed_full']
            observations['full_repeat_vs_b8']=characterize_case(full['cases'][0],prior_a['cases'][0],[0,1,2,3],read_full,read_a)
            assert observations['full_repeat_vs_b8']==partial['full_case_vs_b8']
            for label in ('empty_state','final_reset'):
                assert set(report[label])=={'0'};capture=report[label]['0'];logical=capture['logical']
                assert logical['physical_words_per_row']==106823680 and logical['reference_bf16_roundtrip_checked'] is False and logical['slots']==[0]*4
                assert compare_capture(capture,prior_a[label],[0,1,2,3],read,read_a)['all_states_and_vectors_exact'] is True
            resource=doc(base+'packed-resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'packed-resources/resource-vram.jsonl').splitlines()]
            assert type(resource['child_exit_code']) is int and resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid']
            gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])];assert samples and all(v>0 for v in gaps)
            memory=resource['sampled_memory'];assert memory['samples']==len(samples) and memory['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in samples)
            assert memory['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None) and memory['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None)
            for name in ('cpu-ledger','packed-gpu-ledger'):
                events=[json.loads(line) for line in z.read(base+name+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int and events[1]['exit_code']==0
            assert ('Ran '+str(p['expected_cpu_test_count'])+' tests').encode() in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        result={'status':'b4_packed_cache_characterization_replayed' if passed else 'b4_packed_cache_characterization_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'observations':observations,'correctness_sampled_memory':memory,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(METHODS+Path(__file__).name)),'original_full_cache_gate_changed':False,'cause_established':False,'harmlessness_or_equivalence_proven':False,'b8_mtp_or_http_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    finally:
        z.close()
        for value in previous.values():value.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'observed_exact_pairs':{k:v['all_observed_ids_states_vectors_exact'] for k,v in observations.items()}}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
