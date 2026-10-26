"""Independent complete word/quantization and actual helper/wire archive replay."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fp8_m128_native_control import wire_contract,native_log,replay
JOB='fp8-m128-native-control-attempt1';PARENT='prefill128-fp8-workspace8-attempt1';METHODS='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def checked(path):
    z=zipfile.ZipFile(path);manifest=json.loads(z.read('evidence-manifest.json'))
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
    for name,h in manifest['files'].items():assert sha(z.read(name))==h
    return z,manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);prior,pm=checked(args.proofs/(PARENT+'-evidence.zip'))
    try:
        base='results/'+JOB+'/';fixture='results/actual-layer0-fp8-fixture/';doc=lambda name:json.loads(z.read(name));raw=z.read('results/fp8-m128-native-control-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is True
        assert w['source_before']==w['source_after']==p['files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['gpu_model_executed'] is w['application_library_loaded'] is w['performance_admitted'] is False
        assert w['parent']==p['parent']==manifest['linked_parent_archive'] and sha((args.proofs/(PARENT+'-evidence.zip')).read_bytes())==p['parent']['archive_sha256']
        proof=z.read('results/'+PARENT+'-verified.json');assert proof==(args.proofs/(PARENT+'-verified.json')).read_bytes() and sha(proof)==p['parent']['proof_sha256'] and json.loads(proof)['archive_sha256']==p['parent']['archive_sha256']
        for name,h in p['methods'].items():assert sha(z.read(name))==h==p['files'][name]
        for name in ('fp8_m128_native_control.py','prefill128_fp8_projection.py','verify_fp8_m128_native_control.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][METHODS+name]
        origin=doc(METHODS+'evidence/fp8-m128-native-control-origin.json')
        assert len(origin['candidate_copies'])==5
        for name,row in origin['candidate_copies'].items():assert sha(z.read(METHODS+name))==row['sha256']==p['methods'][METHODS+name] and row['production_source'].startswith('src/backends/tensorrt/')
        record=doc(METHODS+'evidence/fp8-m128-native-control-derivation.json');row=record['files'][0];source=prior.read(METHODS+row['source']);assert sha(source)==row['source_sha256'];text=source.decode()
        for step in row['replacements']:assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
        assert text.encode()==z.read(METHODS+row['target']) and sha(text.encode())==row['target_sha256']
        assert z.read(METHODS+'fp8_m128_observed_helper.cpp')==record['observer_wrapper'].encode() and sha(record['observer_wrapper'].encode())==record['observer_wrapper_sha256']
        for name in ('src/backends/tensorrt/modelopt_cublaslt.cpp','src/backends/tensorrt/modelopt_cublaslt.h','src/backends/tensorrt/modelopt_fp8_plugin.cpp','src/backends/tensorrt/modelopt_fp8_plugin.h','src/cuda/modelopt_fp8.cu','src/cuda/modelopt_fp8.cuh'):assert sha(z.read('Garnet/'+name))==p['files']['Garnet/'+name]
        fraw=z.read(fixture+'manifest.json');f=json.loads(fraw);assert sha(fraw)==p['fixture_manifest_sha256'] and fraw==prior.read(fixture+'manifest.json') and f['files']==p['fixture_files']
        assert w['fixture_after']=={name:row['sha256'] for name,row in f['files'].items()} and manifest['external_weight']==f['files']['weight-n5120-k6144.e4m3']
        for name,row in f['files'].items():
            if name!='weight-n5120-k6144.e4m3':assert len(z.read(fixture+name))==row['bytes'] and sha(z.read(fixture+name))==row['sha256'] and z.read(fixture+name)==prior.read(fixture+name)
        passed=w['exit_code']==0;words={};algorithms={};wire=None
        if passed:
            assert w['status']=='fp8_m128_actual_helper_collected_needs_replay' and sha(z.read(base+'build/control.exe'))==w['executable_sha256']
            for phase in ('cpu','wire-cpu','build','control','native'):
                events=[json.loads(line) for line in z.read(base+phase+'-ledger/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and type(events[1]['exit_code']) is int and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files']
            assert b'Ran 8 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            wire=wire_contract(doc(base+'wire-cpu/report.json'));assert wire==w['wire_contract']
            for mode in ('control','native'):
                assert w[mode+'_exit_code']==0;algorithms[mode]=native_log(z.read(base+mode+'-ledger/commands.log').decode(),mode);assert algorithms[mode]==w[mode+'_algorithms']
                previous='results/'+PARENT+'/'+('shared' if mode=='native' else 'control')+'/'
                words[mode]=replay(lambda name:z.read(fixture+name),lambda name:z.read(base+mode+'/'+name),mode,lambda name:prior.read(previous+name));assert words[mode]==w[mode+'_replay']
            assert algorithms['control']['algorithms'][0]==algorithms['native']['algorithms'][0]
        result={'status':'fp8_m128_actual_native_helper_and_wire_replayed' if passed else 'fp8_m128_actual_native_control_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'wire_contract':wire,'algorithm_observations':algorithms,'complete_output_word_replays':words,'total_output_words_replayed':sum(value['complete_bf16_output_words_replayed'] for value in words.values()),'source_runtime_environment_fixture_unchanged':True,'gpu_model_executed':False,'full_model_mismatch_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    finally:z.close();prior.close()
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'output_words_replayed':result['total_output_words_replayed']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
