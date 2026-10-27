"""Full raw stage replay; passing observations do not admit the known failed model."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
from test2026.cuda.qwen38_5090.full_chunk_stage_replay import replay
JOB='full-chunk-stage-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/full-chunk-stage-gpu-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json');passed=w['exit_code']==0
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert p['installed_files']=={**p['files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}}
        if 'installed_source_sha256' in w:assert w['installed_source_sha256']==p['installed_files']
        else:assert not passed
        if 'parent_installed_source_sha256' in w:assert w['parent_installed_source_sha256']==p['parent_installed_files']
        else:assert not passed
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==51 and w['candidate_library_sha256']==p['candidate_library_sha256']=='3574f8cc1f772933f117b484d9a57844fb3846a4e1ada59f62399876c31fa48a'
        for n,h in {**p['new_files'],**p['audit_dependencies'],**p['checkpoints']}.items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files'][P+Path(__file__).name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(job+'-verified.json')).read_bytes()
            assert q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==51 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
            expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert len(expected)==14 and old.testzip() is None and {n:sha(old.read(n)) for n in old.namelist()}==expected
        def command(phase,required,expected):
            name=base+phase+'-ledger/commands.jsonl'
            if name not in manifest['files']:assert not required;return None
            events=[json.loads(line) for line in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected
            if required:assert events[1]['exit_code']==0
            return events[1]['exit_code']
        cpu_exit=command('cpu',False,p['parent_installed_files']);assert type(cpu_exit) is int
        if cpu_exit==0:assert b'Ran 46 tests' in z.read(base+'cpu-ledger/commands.log') and b'Ran 86 tests' in z.read(base+'cpu-ledger/commands.log')
        else:assert not passed
        if base+'embedded-metadata.json' in manifest['files']:
            m=doc(base+'embedded-metadata.json');assert m['status']=='embedded_full_chunk_stage_metadata_passed' and m['numpy_imported'] is m['GPU_or_Garnet_requested'] is False;command('embedded-metadata',True,p['installed_files'])
        reports={};reads={}
        for phase in ('control32','control128','stages32','stages128'):
            exit_code=command(phase,passed,p['installed_files']);name=base+phase+'/report.json'
            if name in manifest['files']:
                r=doc(name);reports[phase]=r;reads[phase]=lambda n,phase=phase:z.read(base+phase+'/'+n);assert r['candidate_library_sha256']==p['candidate_library_sha256']
            resource=base+phase+'-resources/resource-monitor.json'
            if resource in manifest['files']:m=doc(resource);assert m['gpu']['uuid']==p['gpu_uuid'] and m['child_exit_code']==exit_code
        arrays={'status':'failed_stage_or_primary_capture_preserved_no_full_model_admission','reports':{k:dict(status=r['status'],error=r.get('error'),stage_frames=len(r.get('stage_frames',[]))) for k,r in reports.items()}}
        if passed:
            arrays=replay(reports,reads,args.proofs);assert w['status']=='full_chunk_stages_collected_same_profile_equivalence_needs_replay' and arrays==doc(base+'stage-analysis.json')
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='actual_full_chunk_stage_and_primary_equivalence_replayed' if passed else 'actual_full_chunk_stage_failure_preserved',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),candidate_library_sha256=p['candidate_library_sha256'],source_library_runtime_SDK_checkpoint_restored=True,arrays=arrays,actual_Garnet_frontend_negative_cases_pending=True,full_six_case_model_quality_performance_still_required=True,model_or_performance_qualified=False,two_times_target_achieved=False)
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),arrays=arrays)))
if __name__=='__main__':main()
