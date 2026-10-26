"""Replay actual B4 cache-mode triangle; no tolerance relaxation or quality admission."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.b4_cache_mode_characterization import characterize_case
from test2026.cuda.qwen38_5090.bf16_batch8_external_reference import capture_exact
JOB='b4-cache-mode-characterization-attempt1';A='paged-batch8-model-preservation-attempt1';B='bf16-batch8-external-reference-attempt1'
METHODS='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);manifest=json.loads(z.read('evidence-manifest.json'))
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
    for name,digest in manifest['files'].items():assert sha(z.read(name))==digest
    return z,manifest

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);old,_=checked(args.proofs/(A+'-evidence.zip'));failed,_=checked(args.proofs/(B+'-evidence.zip'))
    observations={};resources={}
    with z,old,failed:
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name))
        w=doc(base+'workflow.json');praw=z.read(METHODS+'evidence/b4-cache-mode-characterization-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['library_unchanged'] is True
        assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime']
        assert w['environment_before']==w['environment_after']==p['environment'] and w['parents']==p['parents']
        for name,digest in p['new_files'].items():assert sha(z.read(name))==digest
        fw=json.loads(failed.read('results/'+B+'/workflow.json'));old_praw=failed.read(METHODS+'evidence/bf16-batch8-external-reference-preparation.json')
        assert p['files']=={**fw['source_after_sha256'],METHODS+'evidence/bf16-batch8-external-reference-preparation.json':sha(old_praw),**p['new_files']}
        assert fw['exit_code']==1 and fw['cleanup_finished'] is fw['library_unchanged'] is True
        assert fw['source_sha256']==fw['source_after_sha256'] and fw['runtime_dependencies']==fw['runtime_dependencies_after']==p['runtime']
        assert fw['environment_before']==fw['environment_after']==p['environment']
        for job,previous in ((A,old),(B,failed)):
            row=p['parents'][job];raw=z.read('results/'+job+'-verified.json');q=json.loads(raw)
            assert raw==(args.proofs/(job+'-verified.json')).read_bytes() and sha(raw)==row['proof_sha256']
            assert q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
            assert manifest['linked_parent_archives'][job]==row['archive_sha256']
            previous_w=json.loads(previous.read('results/'+job+'/workflow.json'))
            assert type(previous_w['exit_code']) is int and previous_w['exit_code']==row['exit_code'] and previous_w['cleanup_finished'] is True
        for name in ('b4_cache_mode_characterization.py','verify_b4_cache_mode_characterization.py','bf16_batch8_external_reference.py','bf16_batch8_reference_gate.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][METHODS+name]
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='b4_cache_mode_characterization_collected_needs_replay' and w['last_completed_mode']=='packed'
            prior_a=json.loads(old.read('results/'+A+'/model/report.json'));prior_b=json.loads(failed.read('results/'+B+'/model/report.json'))
            assert len(prior_a['cases'])==9 and len(prior_b['cases'])==1
            read_a=lambda name:old.read('results/'+A+'/model/'+name)
            read_b=lambda name:failed.read('results/'+B+'/model/'+name)
            reports={}
            for mode in ('full','packed'):
                report=doc(base+mode+'/report.json');reports[mode]=report
                assert report['status']=='b4_cache_mode_characterization_collected_needs_replay' and report['mode']==mode
                assert (report['physical_batch'],report['capacity'],report['prefill_chunk'],report['decode_chunk'])==(4,4096,32,4)
                assert report['bf16_kv_float_reference'] is (mode=='full') and report['compact_commit_used'] is (mode=='packed')
                assert report['mtp_executed'] is report['http_executed'] is report['performance_admitted'] is False
                assert report['library_sha256']==p['runtime']['garnet.dll'] and report['baseline_archive_sha256']==p['parents'][A]['archive_sha256']
                assert len(report['cases'])==report['observed_case_count']==1
                case=report['cases'][0];assert case['physical_rows']==[0,1,2,3] and case['baseline_case_index']==case['half']==0
                read=lambda name:z.read(base+mode+'/'+name)
                comparison=characterize_case(case,prior_a['cases'][0],[0,1,2,3],read,read_a)
                assert comparison==case['characterization'] and case['all_reference_rows_exact'] is comparison['all_observed_ids_states_vectors_exact']
                observations[mode+'_vs_b8']=comparison
                for key in ('empty_state','final_reset'):
                    assert set(report[key])=={'0'}
                    capture_exact(report[key]['0'],prior_a[key],[0,1,2,3],read,read_a)
                resource=doc(base+mode+'-resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+mode+'-resources/resource-vram.jsonl').splitlines()]
                assert type(resource['child_exit_code']) is int and resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid']
                gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])];assert samples and all(v>0 for v in gaps)
                memory=resource['sampled_memory'];assert memory['samples']==len(samples) and memory['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in samples)
                assert memory['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None) and memory['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None)
                resources[mode]=memory
            observations['full_repeat_vs_failed_full']=characterize_case(reports['full']['cases'][0],prior_b['cases'][0],[0,1,2,3],lambda name:z.read(base+'full/'+name),read_b)
            observations['packed_vs_full_repeat']=characterize_case(reports['packed']['cases'][0],reports['full']['cases'][0],[0,1,2,3],lambda name:z.read(base+'packed/'+name),lambda name:z.read(base+'full/'+name))
            for name in ('cpu-ledger','full-gpu-ledger','packed-gpu-ledger'):
                rows=[json.loads(line) for line in z.read(base+name+'/commands.jsonl').splitlines()]
                assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished' and type(rows[1]['exit_code']) is int and rows[1]['exit_code']==0
            assert ('Ran '+str(p['expected_cpu_test_count'])+' tests').encode() in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        result={'status':'b4_cache_mode_characterization_replayed' if passed else 'b4_cache_mode_characterization_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'observations':observations,'correctness_sampled_memory':resources,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(METHODS+Path(__file__).name)),'cause_established':False,'harmlessness_or_equivalence_proven':False,'b8_mtp_or_http_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'observed_exact_pairs':{k:v['all_observed_ids_states_vectors_exact'] for k,v in observations.items()}}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
