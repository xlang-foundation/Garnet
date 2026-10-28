"""Replay actual full4K T128 model evidence; preserve all strict failures."""
from pathlib import Path
import argparse,hashlib,io,json,statistics,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.prefill128_fp8_model import replay_prefill128_fp8_model as replay_prefill128_model,validate_selection
JOB='bf16-prefill128-fp8-model-admission-attempt1';BUILD='bf16-prefill128-fp8-native-build-attempt1';PARENT='mtp-prefill16-full4k-admission-attempt1';METHODS='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);manifest=json.loads(z.read('evidence-manifest.json'))
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
    for name,h in manifest['files'].items():assert sha(z.read(name))==h
    return z,manifest

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);parents={};result=None
    try:
        base='results/'+JOB+'/';doc=lambda name:json.loads(z.read(name));raw=z.read(METHODS+'evidence/bf16-prefill128-fp8-model-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int
        assert w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['candidate_library_sha256']==p['candidate_library_sha256'] and w['baseline_library_sha256']==p['runtime']['garnet.dll'] and w['parents']==p['parents']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==29
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['status']==row['status']
            assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            path=args.proofs/(job+'-evidence.zip');assert sha(path.read_bytes())==row['archive_sha256'];parents[job]=checked(path)[0]
            closed=json.loads(parents[job].read('results/'+job+'/workflow.json'));assert closed['exit_code']==row['exit_code'] and closed['cleanup_finished']
        built=parents[BUILD];old=parents[PARENT];braw=built.read(METHODS+'evidence/bf16-prefill128-fp8-native-build-preparation.json');bp=json.loads(braw)
        inherited={**bp['base_files'],**bp['methods'],METHODS+'evidence/bf16-prefill128-fp8-native-build-preparation.json':sha(braw)}
        assert p['files']=={**inherited,**p['new_files']} and p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
        assert w.get('installed_source_sha256')==p['installed_files']
        for name,h in p['new_files'].items():assert sha(z.read(name))==h
        for name in ('prefill128_model.py','prefill128_fp8_model.py','verify_bf16_prefill128_fp8_model.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['new_files'][METHODS+name]
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==29 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
            for name,h in bp['candidate_files'].items():assert p['candidate_files'][name]==h
            for record in ('prefill128-python-runtime-derivation.json','prefill128-model-source-derivation.json'):
                for row in doc(METHODS+'evidence/'+record)['files']:
                    target=row['target'];digest=row['target_sha256']
                    if target.startswith('xModel/'):assert p['candidate_files'][target]==digest
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None}
            assert len(expected)==12 and original.testzip() is None and len(original.namelist())==len(set(original.namelist()))==len(expected) and set(original.namelist())==set(expected)
            for name,h in expected.items():assert sha(original.read(name))==h==p['files']['Garnet/'+name]
        off_selection=None;selection=None
        if base+'selection-off/report.json' in manifest['files']:
            off=doc(base+'selection-off/report.json');assert off['library_sha256']==p['candidate_library_sha256'] and off['fp8_m128_policy']=='0'
            if off['status']=='fp8_m128_selection_off_constructed':
                assert off['cases']==off['ordinary_observations']==[]
                from test2026.cuda.qwen38_5090.prefill128_fp8_model import validate_targets
                off_selection=validate_targets(off['fp8_target_graphs'],'0');assert off_selection==w['selection_off_counts']
                resource=doc(base+'selection-off-resources/resource-monitor.json');assert resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid']
        passed=w['exit_code']==0;replay=None;memory=None;observed=0
        if base+'native/report.json' in manifest['files']:
            report=doc(base+'native/report.json');assert report['library_sha256']==p['candidate_library_sha256'] and report['quality_qualified'] is report['performance_admitted'] is False
            observed=len(report.get('ordinary_observations',[]))
            if 'fp8_auxiliary_graphs' in report:
                try:selection=validate_selection(report)
                except AssertionError:selection=None
        if passed:
            assert w['status']=='bf16_prefill128_fp8_full4k_model_collected_needs_replay' and off_selection=={'4':0,'128':0} and selection is not None
            reference=json.loads(old.read('results/'+PARENT+'/native/report.json'));cases=json.loads(old.read(METHODS+'evidence/native-host-stage-cases.json'))['cases']
            replay=replay_prefill128_model(report,lambda name:z.read(base+'native/'+name),reference,lambda name:old.read('results/'+PARENT+'/native/'+name),cases)
            assert replay==report['local_replay'] and report['baseline_full4k_archive_sha256']==p['baseline_archive_sha256']
            for name in ('cpu-ledger','selection-off-ledger','native-ledger'):
                events=[json.loads(line) for line in z.read(base+name+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            assert ('Ran '+str(p['expected_cpu_test_count'])+' tests').encode() in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            resource=doc(base+'native-resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'native-resources/resource-vram.jsonl').splitlines()]
            assert type(resource['child_exit_code']) is int and resource['child_exit_code']==0 and resource['gpu']['uuid']==p['gpu_uuid'] and samples
            gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])];assert all(value>0 for value in gaps)
            memory=resource['sampled_memory'];assert memory['samples']==len(samples) and memory['sampled_peak_adapter_used_bytes']==max(value['used_bytes'] for value in samples)
            assert memory['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None) and memory['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None)
        result={'status':'bf16_prefill128_fp8_full4k_model_replayed' if passed else 'bf16_prefill128_fp8_full4k_model_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':p['candidate_library_sha256'],'baseline_library_sha256':p['runtime']['garnet.dll'],'source_runtime_environment_restored':True,'ordinary_observations_collected':observed,'default_off_selection':off_selection,'enabled_native_selection':selection,'model_replay':replay,'correctness_sampled_memory':memory,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:
        z.close()
        for value in parents.values():value.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'ordinary_observations_collected':result['ordinary_observations_collected']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
