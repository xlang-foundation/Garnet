"""Replay observed actual helper choices and restored same-DLL diagnostic transaction."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fp8_z128_context_audit import replay
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import complete_entries
JOB='fp8-z128-context-audit-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        base='results/'+JOB+'/';praw=z.read(PREFIX+'evidence/fp8-z128-context-audit-preparation.json');p=json.loads(praw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(praw)==w['preparation_sha256'] and w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        assert len(p['candidate_files'])==28 and w['candidate_library_sha256']==p['candidate_library_sha256'] and w['policy_controls']==[]
        for name,h in p['new_files'].items():assert sha(z.read(name))==h
        for name,h in p['observed_helper_files'].items():assert sha(z.read(name))==h
        for name in ('verify_fp8_z128_context_audit.py','fp8_z128_context_audit.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['new_files'][PREFIX+name]
        for parent,row in p['parents'].items():
            raw=z.read('results/'+parent+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][parent]
            assert raw==(args.proofs/(parent+'-verified.json')).read_bytes() and sha((args.proofs/(parent+'-evidence.zip')).read_bytes())==row['archive_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as bundle:
            assert bundle.testzip() is None and set(bundle.namelist())==set(p['candidate_files']) and len(bundle.namelist())==28
            for name,h in p['candidate_files'].items():assert sha(bundle.read(name))==h
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        for name,h in p['checkpoints'].items():assert sha(z.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(expected)==8 and set(old.namelist())==set(expected)
            for name,h in expected.items():assert sha(old.read(name))==h
        assert b'Ran 37 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        for phase in ('cpu-ledger','audit-build-ledger','cold-ledger','native-ledger'):
            events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            assert events[1]['exit_code']==(1 if phase=='native-ledger' else 0)
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
        for phase in ('cold','native'):
            resource=json.loads(z.read(base+phase+'-resources/resource-monitor.json'));assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==(1 if phase=='native' else 0)
        receipt=json.loads(z.read(base+'audit-library.json'));assert receipt['library_sha256']==w['audit_library_sha256']==sha(z.read(base+'build/context-audit.dll')) and receipt['cublaslt_sha256']==p['runtime']['cublasLt64_13.dll'] and receipt['expected_m32_opaque']==p['expected_m32_opaque']
        report=json.loads(z.read(base+'native/report.json'));assert report['candidate_library_sha256']==p['candidate_library_sha256'] and report['error']=="RuntimeError('TensorRT buildSerializedNetwork failed')" and [(r['layer'],r['tokens'],r['diagnostic']) for r in report['cases']]==[(0,32,False),(0,32,True)] and 'context_audit_error' not in report
        assert len(report['frontend_rejections'])==10 and all(r['rejected'] for r in report['frontend_rejections'])
        with zipfile.ZipFile(args.proofs/'fp8-z128-stage-witness-attempt1-evidence.zip') as parent:
            raw=parent.read(PREFIX+'evidence/native-host-stage-cases.json');assert sha(raw)==p['files'][PREFIX+'evidence/native-host-stage-cases.json'] and report['input_ids']==json.loads(raw)['cases'][0]['prompt_ids'][:128]
        observations={}
        cold=json.loads(z.read(base+'cold/report.json'));assert cold==w['cold_context_observation']
        for phase,bridge in (('cold',cold['observation']),('native',report['post_failure_context_audit'])):
            raw=z.read(base+phase+'/context-audit.json');observed=json.loads(raw);validated=replay(observed,p['expected_m32_opaque'])
            assert validated==bridge['validation'] and sha(raw)==bridge['report_sha256'] and bridge['library_sha256']==receipt['library_sha256'] and bridge['loaded_cublaslt_sha256']==receipt['cublaslt_sha256']
            observations[phase]={'replay':validated,'report_sha256':sha(raw),'actual_candidates':observed['heuristics'][0]['candidates'],'helper_info':observed['helper_info']}
        entries=complete_entries(report,lambda name:z.read(base+'native/'+name));assert entries==203
        result={'status':'fp8_z128_context_audit_and_partial_main_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':p['candidate_library_sha256'],'observer_library_sha256':receipt['library_sha256'],'workflow_exit_code':1,'profiles_observed':2,'complete_finite_array_entries':entries,'context_observations':observations,'source_runtime_sdk_checkpoint_restored':True,'full_Garnet_plugin_failure_directly_instrumented':False,'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'cold':observations['cold']['replay'],'post_failure':observations['native']['replay']}));return 0

if __name__=='__main__':raise SystemExit(main())
