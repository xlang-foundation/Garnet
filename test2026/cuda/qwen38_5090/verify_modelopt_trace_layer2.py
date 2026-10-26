"""Replay actual plugin initialization events, retained controls and full source restoration."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
from test2026.cuda.qwen38_5090.modelopt_trace_layer2 import trace_profiles,compare_profiles,stable_array_replay
from test2026.cuda.qwen38_5090.fp8_z128_context_audit import replay as audit_replay
JOB='modelopt-trace-layer2-attempt1';OLD='layer2-ordinary-witness-attempt1';STAGE='fp8-z128-native-package-audit-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={}
    try:
        base='results/'+JOB+'/';praw=z.read(M+'evidence/modelopt-trace-layer2-preparation.json');p=json.loads(praw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and w['candidate_library_sha256']==p['candidate_library_sha256'] and w['installed_source_sha256']==p['installed_files']
        for name,h in {**p['new_files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(z.read(name))==h
        for name in ('modelopt_trace_layer2.py','verify_modelopt_trace_layer2.py','modelopt_init_trace_contract.py','layer2_ordinary_replay.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][M+name]
        for parent,row in p['parents'].items():
            proof=z.read('results/'+parent+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][parent]
            assert proof==(a.proofs/(parent+'-verified.json')).read_bytes() and sha((a.proofs/(parent+'-evidence.zip')).read_bytes())==row['archive_sha256'];parents[parent]=checked(a.proofs/(parent+'-evidence.zip'))[0]
        source_raw=z.read(p['source_bundle']);assert sha(source_raw)==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(source_raw)) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==38 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
        saved={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(saved)==14
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            assert original.testzip() is None and len(original.namelist())==len(set(original.namelist()))==14 and set(original.namelist())==set(saved)
            for name,h in saved.items():assert sha(original.read(name))==h==p['files']['Garnet/'+name]
        oldbase='results/'+OLD+'/native/';prior=json.loads(parents[OLD].read(oldbase+'report.json'));common=z.read(M+'evidence/layer2-ordinary-common-input.f32');assert sha(common)==p['common_input_sha256']
        assert b'Ran 83 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        phases={};reports={}
        for phase in ('cpu','trace-on','trace-off'):
            key=base+phase+'-ledger/commands.jsonl'
            if key not in manifest['files']:continue
            ledger=[json.loads(raw) for raw in z.read(key).splitlines()];assert len(ledger)==2 and ledger[0]['event']=='started' and ledger[1]['event']=='finished' and ledger[0]['source']['owned_source_sha256']==ledger[1]['source_after']['owned_source_sha256']==p['installed_files']
            if phase=='cpu':assert ledger[1]['exit_code']==0;continue
            r=json.loads(z.read(base+phase+'/report.json'));reports[phase]=r;assert r['candidate_library_sha256']==p['candidate_library_sha256'] and r['input_ids']==p['common_input_metadata']['input_ids'] and r['trace_policy']==('1' if phase=='trace-on' else None)
            done=r['status']=='layer2_ordinary_collected_needs_replay';resource=json.loads(z.read(base+phase+'-resources/resource-monitor.json'));assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==ledger[1]['exit_code'] and (not done or ledger[1]['exit_code']==0)
            trace=trace_profiles(z.read(base+phase+'-ledger/commands.log'),phase=='trace-on',done);read=lambda name:z.read(base+phase+'/'+name);row={'status':r['status'],'error':r.get('error'),'trace':trace,'array_replay':stable_array_replay(r,read,common) if done else None,'retained_T32_comparison':None,'postfailure_helper':None}
            if any(c['tokens']==32 and len(c['repeats'])==2 and all(len(s)==4 for s in c['repeats']) for c in r['cases']):row['retained_T32_comparison']=compare_profiles(r,prior,read,lambda name:parents[OLD].read(oldbase+name),(32,));assert row['retained_T32_comparison']['complete_finite_entries_compared']==58
            if 'initialization_context_audit' in r:
                raw=z.read(base+phase+'/context-audit.json');receipt=json.loads(z.read('results/'+STAGE+'/audit-library.json'));row['postfailure_helper']=audit_replay(json.loads(raw),receipt['expected_m32_opaque']);assert sha(raw)==r['initialization_context_audit']['report_sha256'] and row['postfailure_helper']==r['initialization_context_audit']['validation']
            phases[phase]=row
        paired=None;passed=w['exit_code']==0
        if passed:
            assert set(phases)=={'trace-on','trace-off'} and all(row['array_replay'] is not None for row in phases.values())
            paired=compare_profiles(reports['trace-on'],reports['trace-off'],lambda name:z.read(base+'trace-on/'+name),lambda name:z.read(base+'trace-off/'+name),(32,128));assert paired['complete_finite_entries_compared']==73
        failed_routes=phases.get('trace-on',{}).get('trace',{}).get('failed_initialize_returns',[])
        result={'status':'actual_modelopt_layer2_trace_and_controls_replayed' if passed else 'actual_modelopt_layer2_initialization_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':p['candidate_library_sha256'],'source_runtime_SDK_checkpoint_restored':True,'phases':phases,'trace_on_off_array_parity':paired,'actual_failed_initialize_returns':failed_routes,'actual_failed_plugin_route_observed':bool(failed_routes),'helper_API_failure_cause_established':False,'initialization_reliability_resolved':False,'full_model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for value in parents.values():value.close()
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'failed_initialize_returns':failed_routes}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
