"""Verify immutable same-DLL trace and restore transaction; never admit performance."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import complete_entries
from test2026.cuda.qwen38_5090.fp8_z128_stage_replay import replay
JOB='fp8-z128-stage-trace-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'));assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        base='results/'+JOB+'/';praw=z.read(PREFIX+'evidence/fp8-z128-stage-trace-preparation.json');p=json.loads(praw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(praw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        assert len(p['candidate_files'])==28 and w['candidate_library_sha256']==p['candidate_library_sha256'] and w['policy_controls']==[] and 'Ten controls closed' in w['fresh_policy_controls_omitted']
        for name,h in p['new_files'].items():assert sha(z.read(name))==h
        for name in ('verify_fp8_z128_stage_trace.py','fp8_z128_stage_replay.py','fp8_z128_stage_layout.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][PREFIX+name]
        for parent,row in p['parents'].items():
            raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw);assert sha(raw)==row['proof_sha256'] and proof['status']==row['status'] and proof['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][parent]
            assert raw==(args.proofs/(parent+'-verified.json')).read_bytes() and sha((args.proofs/(parent+'-evidence.zip')).read_bytes())==row['archive_sha256']
        derivation=json.loads(z.read(PREFIX+'evidence/fp8-z128-stage-trace-derivation.json'))
        assert not derivation['native_or_collector_modified'] and len(derivation['derivations'])==2
        with zipfile.ZipFile(args.proofs/'fp8-z128-stage-witness-attempt1-evidence.zip') as failed:
            for row in derivation['derivations']:
                raw=failed.read(PREFIX+row['base']);assert sha(raw)==row['base_sha256']==p['files'][PREFIX+row['base']]
                for change in row['changes']:
                    before=change['before'].encode();assert raw.count(before)==1;raw=raw.replace(before,change['after'].encode())
                assert raw==z.read(PREFIX+row['target']) and sha(raw)==row['target_sha256']
        for name,h in p['checkpoints'].items():assert sha(z.read(name))==h
        source=z.read(p['source_bundle']);assert sha(source)==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(source)) as bundle:
            assert bundle.testzip() is None and len(bundle.namelist())==28 and set(bundle.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(bundle.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(expected)==8 and set(old.namelist())==set(expected)
            for name,h in expected.items():assert sha(old.read(name))==h
        assert b'Ran 32 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        for phase in ('cpu-ledger','native-ledger'):
            events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
        resource=json.loads(z.read(base+'native-resources/resource-monitor.json'));assert resource['gpu']['uuid']==p['gpu_uuid']
        log=z.read(base+'native-cublaslt.log');assert len(log)>0 and b'cublasLt' in log
        with zipfile.ZipFile(args.proofs/'fp8-z128-stage-witness-attempt1-evidence.zip') as linked:
            case_raw=linked.read(PREFIX+'evidence/native-host-stage-cases.json');assert sha(case_raw)==p['files'][PREFIX+'evidence/native-host-stage-cases.json']
        report=json.loads(z.read(base+'native/report.json'));assert report['candidate_library_sha256']==p['candidate_library_sha256'] and report['input_ids']==json.loads(case_raw)['cases'][0]['prompt_ids'][:128]
        entries=complete_entries(report,lambda name:z.read(base+'native/'+name));full=None
        if report['status']=='gdn_chunk_stage_collected_needs_replay':full=replay(report,lambda name:z.read(base+'native/'+name))
        if w['exit_code']==0:assert full and full['controlled_first128_all11_stages_exact'] and entries==509
        result={'status':'fp8_z128_same_DLL_initialization_trace_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'trace_sha256':sha(log),'trace_bytes':len(log),'candidate_library_sha256':p['candidate_library_sha256'],'workflow_exit_code':w['exit_code'],'native_exit_code':resource['child_exit_code'],'report_status':report['status'],'report_error':report.get('error'),'profiles_observed':len(report['cases']),'complete_finite_array_entries':entries,'full_stage_replay':full,'source_runtime_sdk_checkpoint_restored':True,'plugin_initialization_root_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'trace_bytes':len(log),'native_exit_code':result['native_exit_code']}));return 0

if __name__=='__main__':raise SystemExit(main())
