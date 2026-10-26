"""Independent complete piece/primary/stage evidence replay; preserve failures."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_chunk_stage_replay import replay,array
JOB='gdn-chunk-stage-witness-attempt2';BUILD='gdn-chunk-stage-native-build-attempt1';METHODS='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);m=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for name,h in m['files'].items():assert sha(z.read(name))==h
    return z,m

def complete_entries(value,read):
    if isinstance(value,dict):
        if set(value)=={'words','sha256','parts'}:array(value,read);return 1
        return sum(complete_entries(item,read) for item in value.values())
    if isinstance(value,list):return sum(complete_entries(item,read) for item in value)
    return 0

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={}
    try:
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name));raw=z.read(METHODS+'evidence/gdn-chunk-stage-witness-bound-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['checkpoint_after']==p['checkpoints']
        for name,h in p['checkpoints'].items():assert sha(z.read(name))==h
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==23 and w['candidate_library_sha256']==p['candidate_library_sha256']
        assert w.get('installed_source_sha256')==p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256'];parents[job]=checked(args.proofs/(job+'-evidence.zip'))[0]
        built=parents[BUILD];braw=built.read(METHODS+'evidence/gdn-chunk-stage-native-build-preparation.json');bp=json.loads(braw)
        failed=parents['gdn-chunk-stage-witness-attempt1'];fraw=failed.read(METHODS+'evidence/gdn-chunk-stage-witness-preparation.json');fp=json.loads(fraw)
        inherited={**fp['files'],METHODS+'evidence/gdn-chunk-stage-witness-preparation.json':sha(fraw)}
        closed=json.loads(failed.read('results/gdn-chunk-stage-witness-attempt1/workflow.json'));assert closed['exit_code']==1 and closed['cleanup_finished'] and closed['baseline_source_restored'] and closed['baseline_library_restored']
        assert p['files']=={**inherited,**p['new_files']} and bp['candidate_files']==p['candidate_files'] and bp['candidate_base']==p['candidate_base'] and bp['runtime']==p['runtime']
        for name,h in p['new_files'].items():assert sha(z.read(name))==h
        for name in ('gdn_chunk_stage_layout.py','gdn_chunk_stage_replay.py','verify_gdn_chunk_stage_witness_bound.py'):assert sha(Path(__file__).with_name(name).read_bytes())==p['new_files'][METHODS+name]
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==23 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(expected)==8 and set(original.namelist())==set(expected)
            for name,h in expected.items():assert sha(original.read(name))==h==p['files']['Garnet/'+name]
        report=doc(base+'native/report.json') if base+'native/report.json' in manifest['files'] else None;entries=0;full=None;passed=w['exit_code']==0
        if report:
            assert report['candidate_library_sha256']==p['candidate_library_sha256'] and report['model_or_performance_qualified'] is False
            assert report['cases_sha256']==p['files'][METHODS+'evidence/native-host-stage-cases.json'] and report['config_sha256']==p['files']['models/Qwen3.8-27B-NVFP4/config.json']
            assert report['input_ids']==doc(METHODS+'evidence/native-host-stage-cases.json')['cases'][0]['prompt_ids'][:128]
            entries=complete_entries(report,lambda name:z.read(base+'native/'+name))
        if passed:
            assert w['status']=='gdn_chunk_stage_witness_collected_needs_replay' and report
            full=replay(report,lambda name:z.read(base+'native/'+name));assert full==w['array_replay']
            for phase in ('cpu-ledger','native-ledger'):
                events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            assert ('Ran '+str(p['expected_cpu_test_count'])+' tests').encode() in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            resource=doc(base+'native-resources/resource-monitor.json');assert resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid']
        result={'status':'gdn_chunk_stage_witness_arrays_replayed' if passed else 'gdn_chunk_stage_witness_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'complete_finite_array_entries_verified':entries,'array_replay':full,'candidate_library_sha256':p['candidate_library_sha256'],'source_runtime_sdk_restored':True,'frontend_rejections_observed':len(report.get('frontend_rejections',[])) if report else 0,'profiles_observed':len(report.get('cases',[])) if report else 0,'report_error':report.get('error') if report else None,'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for parent in parents.values():parent.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'complete_entries':entries,'profiles_observed':result['profiles_observed']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
