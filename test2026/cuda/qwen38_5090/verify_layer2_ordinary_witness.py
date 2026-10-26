"""Full source/restoration and complete real common-input ordinary-layer replay."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
from test2026.cuda.qwen38_5090.layer2_ordinary_replay import replay,extract_common_input
from test2026.cuda.qwen38_5090.fp8_z128_context_audit import replay as audit_replay
JOB='layer2-ordinary-witness-attempt1';STAGE='fp8-z128-native-package-audit-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={}
    try:
        base='results/'+JOB+'/';praw=z.read(M+'evidence/layer2-ordinary-witness-preparation.json');p=json.loads(praw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and w['candidate_library_sha256']==p['candidate_library_sha256'] and w['installed_source_sha256']==p['installed_files'] and p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        for name,h in {**p['new_files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(z.read(name))==h
        for name in ('layer2_ordinary_layout.py','layer2_ordinary_replay.py','verify_layer2_ordinary_witness.py','gdn_chunk_stage_replay.py','fp8_z128_stage_layout.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][M+name]
        for parent,row in p['parents'].items():
            proof=z.read('results/'+parent+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][parent]
            assert proof==(a.proofs/(parent+'-verified.json')).read_bytes() and sha((a.proofs/(parent+'-evidence.zip')).read_bytes())==row['archive_sha256'];parents[parent]=checked(a.proofs/(parent+'-evidence.zip'))[0]
            closed=json.loads(parents[parent].read('results/'+parent+'/workflow.json'));assert closed['exit_code']==row['exit_code'] and closed['cleanup_finished']
        source_raw=z.read(p['source_bundle']);assert sha(source_raw)==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(source_raw)) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==35 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
        saved={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(saved)==12
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            assert original.testzip() is None and len(original.namelist())==len(set(original.namelist()))==12 and set(original.namelist())==set(saved)
            for name,h in saved.items():assert sha(original.read(name))==h==p['files']['Garnet/'+name]
        stage=parents[STAGE];oldbase='results/'+STAGE+'/native/';old=json.loads(stage.read(oldbase+'report.json'));common=extract_common_input(old,lambda name:stage.read(oldbase+name));info=json.loads(z.read(M+'evidence/layer2-ordinary-common-input.json'))
        assert common==z.read(M+'evidence/layer2-ordinary-common-input.f32') and sha(common)==p['common_input_sha256']==info['sha256'] and info==p['common_input_metadata'] and info['input_ids']==old['input_ids'] and info['archive_sha256']==p['parents'][STAGE]['archive_sha256']
        assert b'Ran 78 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        for phase in ('cpu','native'):
            events=[json.loads(raw) for raw in z.read(base+phase+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            if phase=='cpu' or w['exit_code']==0:assert events[1]['exit_code']==0
        resource=json.loads(z.read(base+'native-resources/resource-monitor.json'));assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==w['exit_code']
        r=json.loads(z.read(base+'native/report.json'));assert r['candidate_library_sha256']==p['candidate_library_sha256'] and r['input_ids']==info['input_ids'] and r['input_fixture_sha256']==p['common_input_sha256'];observation=None
        if 'initialization_context_audit' in r:
            raw=z.read(base+'native/context-audit.json');receipt=json.loads(z.read('results/'+STAGE+'/audit-library.json'));observation=audit_replay(json.loads(raw),receipt['expected_m32_opaque']);bridge=r['initialization_context_audit'];assert sha(raw)==bridge['report_sha256'] and observation==bridge['validation'] and bridge['native_package_library_sha256']==receipt['native_package_library_sha256'] and bridge['library_sha256']==receipt['library_sha256']
        passed=w['exit_code']==0;arrays=replay(r,lambda name:z.read(base+'native/'+name),common) if passed else None
        result={'status':'layer2_ordinary_controlled_arrays_replayed' if passed else 'layer2_ordinary_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':p['candidate_library_sha256'],'common_input_sha256':p['common_input_sha256'],'source_runtime_SDK_checkpoint_restored':True,'complete_array_replay':arrays,'initialization_context_observation':observation,'report_error':r.get('error'),'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for value in parents.values():value.close()
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'arrays':arrays}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
