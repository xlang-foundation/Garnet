"""Full native35/Python model admission replay; success and every strict failure are retained."""
from pathlib import Path
import argparse,hashlib,io,json,statistics,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fp8_z128_full4k_model import replay_full4k_model,validate_selection,validate_targets
from test2026.cuda.qwen38_5090.fp8_z128_context_audit import replay as replay_context
JOB='fp8-z128-full4k-model-admission-attempt1';STAGE='fp8-z128-native-package-audit-attempt1';PARENT='mtp-prefill16-full4k-admission-attempt1';FAILED='bf16-prefill128-fp8-model-admission-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
    for name,h in manifest['files'].items():assert sha(z.read(name))==h
    return z,manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={}
    try:
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name));raw=z.read(M+'evidence/fp8-z128-full4k-model-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['candidate_library_sha256']==p['candidate_library_sha256'] and w['baseline_library_sha256']==p['runtime']['garnet.dll'] and w['parents']==p['parents'] and w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==35
        assert p['files']=={**p['inherited_files'],**p['new_files']} and w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        for name,h in {**p['new_files'],**p['audit_dependencies'],**p['checkpoints']}.items():assert sha(z.read(name))==h
        for name in ('prefill128_model.py','prefill128_fp8_model.py','fp8_z128_full4k_model.py','verify_fp8_z128_full4k_model.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][M+name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            path=args.proofs/(job+'-evidence.zip');assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and sha(path.read_bytes())==row['archive_sha256'];parents[job]=checked(path)[0]
            closed=json.loads(parents[job].read('results/'+job+'/workflow.json'));assert closed['exit_code']==row['exit_code'] and closed['cleanup_finished']
        stage=parents[STAGE];spraw=stage.read(M+'evidence/fp8-z128-native-package-audit-preparation.json');sp=json.loads(spraw)
        oldprep=parents[FAILED].read(M+'evidence/bf16-prefill128-fp8-model-preparation.json');op=json.loads(oldprep);assert sha(oldprep)==p['old_model_preparation_sha256']
        expected={**sp['candidate_files'],**{name:h for name,h in op['candidate_files'].items() if name.startswith('xModel/')}}
        assert expected==p['candidate_files'] and len(sp['candidate_files'])==28 and p['candidate_library_sha256']==json.loads(stage.read('results/'+STAGE+'/workflow.json'))['candidate_library_sha256']
        inherited={**sp['files'],M+'evidence/fp8-z128-native-package-audit-preparation.json':sha(spraw),**p['audit_dependencies']}
        inherited[M+'verify_fp8_z128_native_stage_success.py']=p['stage_success_verifier_sha256'];assert inherited==p['inherited_files']
        expected_tests={name for name in op['test_ids'] if name not in p['superseded_source_test_ids']}|set(p['composed_native_test_ids'])|set(p['new_model_test_ids'])
        assert len(op['test_ids'])==64 and len(p['superseded_source_test_ids'])==3 and all('.test_fp8_m128_native_source.' in name for name in p['superseded_source_test_ids']) and len(p['composed_native_test_ids'])==5 and len(p['new_model_test_ids'])==4 and len(expected_tests)==len(p['test_ids'])==70 and expected_tests==set(p['test_ids'])
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==35 and set(source.namelist())==set(expected)
            for name,h in expected.items():assert sha(source.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            saved={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(saved)==12 and original.testzip() is None and len(original.namelist())==len(set(original.namelist()))==12 and set(original.namelist())==set(saved)
            for name,h in saved.items():assert sha(original.read(name))==h==p['files']['Garnet/'+name]
        audit_receipt=doc('results/'+STAGE+'/audit-library.json');observations={};off_selection=selection=None;report=None
        for phase in ('selection-off','native'):
            if base+phase+'/report.json' not in manifest['files']:continue
            observed=doc(base+phase+'/report.json');assert observed['library_sha256']==p['candidate_library_sha256'] and observed['quality_qualified'] is observed['performance_admitted'] is False
            if 'initialization_context_audit' in observed:
                bridge=observed['initialization_context_audit'];raw=z.read(base+phase+'/initialization-context-audit.json');audit=json.loads(raw);validated=replay_context(audit,audit_receipt['expected_m32_opaque'])
                assert sha(raw)==bridge['report_sha256'] and validated==bridge['validation'] and bridge['native_package_library_sha256']==audit_receipt['native_package_library_sha256'] and bridge['library_sha256']==audit_receipt['library_sha256'];observations[phase]=validated
            if phase=='selection-off' and observed['status']=='fp8_m128_selection_off_constructed':
                assert observed['fp8_m128_policy']==observed['fp8_z128_policy']=='0' and observed['cases']==observed['ordinary_observations']==[];off_selection=validate_targets(observed['fp8_target_graphs'],'0');assert off_selection==w['selection_off_counts']
            elif phase=='native':report=observed
        passed=w['exit_code']==0;replayed=memory=None;observed_count=len(report.get('ordinary_observations',[])) if report else 0
        if report and 'fp8_auxiliary_graphs' in report:
            try:selection=validate_selection(report)
            except AssertionError:selection=None
        if passed:
            assert w['status']=='fp8_z128_full4k_model_collected_needs_replay' and off_selection=={'m128':{'4':0,'128':0},'z128':{'4':0,'128':0}} and selection is not None and report is not None
            old=parents[PARENT];reference=json.loads(old.read('results/'+PARENT+'/native/report.json'));cases=json.loads(old.read(M+'evidence/native-host-stage-cases.json'))['cases']
            replayed=replay_full4k_model(report,lambda name:z.read(base+'native/'+name),reference,lambda name:old.read('results/'+PARENT+'/native/'+name),cases);assert replayed==report['local_replay'] and report['baseline_full4k_archive_sha256']==p['baseline_archive_sha256']
        for phase in ('cpu','selection-off','native'):
            key=base+phase+'-ledger/commands.jsonl'
            if key not in manifest['files']:continue
            events=[json.loads(line) for line in z.read(key).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            if passed:assert events[1]['exit_code']==0
            if phase!='cpu':
                resource=doc(base+phase+'-resources/resource-monitor.json');assert resource['gpu']['uuid']==p['gpu_uuid']
                if passed:assert resource['child_exit_code']==0
                if phase=='native':
                    samples=[json.loads(line) for line in z.read(base+'native-resources/resource-vram.jsonl').splitlines()];assert samples;gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])];assert all(gap>0 for gap in gaps);memory=resource['sampled_memory'];assert memory['samples']==len(samples) and memory['sampled_peak_adapter_used_bytes']==max(row['used_bytes'] for row in samples) and memory['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None) and memory['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None)
        assert b'Ran 70 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        result={'status':'fp8_z128_full4k_model_replayed' if passed else 'fp8_z128_full4k_model_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':p['candidate_library_sha256'],'source_runtime_SDK_checkpoint_restored':True,'ordinary_observations_collected':observed_count,'default_off_selection':off_selection,'enabled_native_selection':selection,'model_replay':replayed,'initialization_context_observations':observations,'report_error':report.get('error') if report else w.get('error'),'correctness_sampled_memory':memory,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for value in parents.values():value.close()
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'ordinary_observations_collected':observed_count}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
