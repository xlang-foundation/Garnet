"""Replay real Z stage arrays, fresh policy controls and restored native28 source."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import array
from test2026.cuda.qwen38_5090.fp8_z128_stage_replay import replay
from test2026.cuda.qwen38_5090.fp8_z128_stage_layout import POLICY_VALUES,check_policy_report
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import complete_entries
JOB='fp8-z128-stage-witness-attempt1';BUILD='fp8-z128-native-build-attempt1';PREVIOUS='gdn-chunk-stage-witness-attempt2';METHODS='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);m=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for name,h in m['files'].items():assert sha(z.read(name))==h
    return z,m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={}
    try:
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name));praw=z.read(METHODS+'evidence/fp8-z128-stage-witness-preparation.json');p=json.loads(praw);w=doc(base+'workflow.json')
        assert sha(praw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        for name,h in p['checkpoints'].items():assert sha(z.read(name))==h
        assert len(p['candidate_files'])==28 and w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and w['candidate_library_sha256']==p['candidate_library_sha256']
        assert w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        for name,h in p['new_files'].items():assert sha(z.read(name))==h
        for name in ('fp8_z128_stage_layout.py','fp8_z128_stage_replay.py','verify_fp8_z128_stage_witness.py'):assert sha(Path(__file__).with_name(name).read_bytes())==p['new_files'][METHODS+name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256'];parents[job]=checked(args.proofs/(job+'-evidence.zip'))[0]
        built=parents[BUILD];braw=built.read(METHODS+'evidence/fp8-z128-native-build-preparation.json');bp=json.loads(braw);inherited={**bp['base_files'],**bp['methods'],METHODS+'evidence/fp8-z128-native-build-preparation.json':sha(braw)}
        assert p['files']=={**inherited,**p['new_files']} and p['candidate_files']==bp['candidate_files'] and p['candidate_base']==bp['candidate_base'] and p['runtime']==bp['runtime'] and p['environment']==bp['environment']
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']==bp['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==28 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(expected)==8 and set(original.namelist())==set(expected)
            for name,h in expected.items():assert sha(original.read(name))==h==p['files']['Garnet/'+name]
        policies=[]
        for row in w.get('policy_controls',[]):
            index=row['index'];assert type(index) is int and index==len(policies);value=POLICY_VALUES[index];assert row['expected_environment_value']==value
            report=doc(base+'policy-'+str(index)+'/report.json');validated=check_policy_report(report,value);assert validated==row['validated'] and report['candidate_library_sha256']==p['candidate_library_sha256'];policies.append(validated)
            resource=doc(base+'policy-'+str(index)+'-resources/resource-monitor.json');assert resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid']
        report=doc(base+'native/report.json') if base+'native/report.json' in manifest['files'] else None;entries=complete_entries(report,lambda name:z.read(base+'native/'+name)) if report else 0;full=None;baseline_arrays=0
        if report:
            assert report['candidate_library_sha256']==p['candidate_library_sha256'] and report['config_sha256']==p['checkpoints']['models/Qwen3.8-27B-NVFP4/config.json'] and report['cases_sha256']==p['files'][METHODS+'evidence/native-host-stage-cases.json']
            assert report['input_ids']==doc(METHODS+'evidence/native-host-stage-cases.json')['cases'][0]['prompt_ids'][:128]
        if report and report['status']=='gdn_chunk_stage_collected_needs_replay':
            full=replay(report,lambda name:z.read(base+'native/'+name));assert full==w['array_replay']
            old=parents[PREVIOUS];oldbase='results/'+PREVIOUS+'/native/';oldreport=json.loads(old.read(oldbase+'report.json'));oldrows={(row['layer'],row['tokens'],row['diagnostic']):row for row in oldreport['cases']}
            assert report['input_ids']==oldreport['input_ids']
            for row in report['cases']:
                if row['tokens']!=32:continue
                prior=oldrows[(row['layer'],32,row['diagnostic'])]
                for sequence,reference in zip(row['repeats'],prior['repeats']):
                    assert len(sequence)==len(reference)==4
                    for now,before in zip(sequence,reference):
                        assert now['start']==before['start'] and now['repeat']==before['repeat']
                        for name in ('input_hidden','input_history','input_state','input_hidden_after','input_history_after','input_state_after'):
                            assert array(now[name],lambda name:z.read(base+'native/'+name))==array(before[name],lambda name:old.read(oldbase+name));baseline_arrays+=1
                        assert set(now['arrays'])==set(before['arrays'])
                        for name in now['arrays']:
                            assert array(now['arrays'][name],lambda name:z.read(base+'native/'+name))==array(before['arrays'][name],lambda name:old.read(oldbase+name));baseline_arrays+=1
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='fp8_z128_stage_witness_collected_needs_replay' and len(policies)==10 and full and full['controlled_first128_all11_stages_exact']
            assert entries==509 and baseline_arrays>0
            phases=['cpu-ledger','native-ledger']+['policy-'+str(index)+'-ledger' for index in range(10)]
            for phase in phases:
                events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            assert b'Ran 29 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            resource=doc(base+'native-resources/resource-monitor.json');assert resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid']
        result={'status':'fp8_z128_stage_policies_primary_and_complete_arrays_replayed' if passed else 'fp8_z128_stage_failure_or_remaining_difference_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'complete_finite_array_entries_verified':entries,'policy_controls':policies,'array_replay':full,'unchanged_T32_baseline_arrays_compared':baseline_arrays,'candidate_library_sha256':p['candidate_library_sha256'],'source_runtime_sdk_checkpoint_restored':True,'frontend_rejections_observed':len(report.get('frontend_rejections',[])) if report else 0,'profiles_observed':len(report.get('cases',[])) if report else 0,'report_error':w.get('error') or (report.get('error') if report else None),'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for parent in parents.values():parent.close()
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'policies':len(policies),'complete_entries':entries,'T32_baseline_arrays':baseline_arrays}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
