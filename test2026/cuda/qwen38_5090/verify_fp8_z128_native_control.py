"""Full typed Z128 helper/plugin/array/source replay; no application-model claim."""
from pathlib import Path,PureWindowsPath
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fp8_z128_native_control import replay_algorithms,replay_arrays
JOB='fp8-z128-native-control-attempt1';PARENT='gdn-z-native-control-attempt1';METHODS='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);m=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for name,h in m['files'].items():assert sha(z.read(name))==h
    return z,m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive)
    with z:
        base='results/'+JOB+'/';w=json.loads(z.read(base+'workflow.json'));praw=z.read('results/fp8-z128-native-control-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] and w['before']==w['after']
        assert w['before']['source']==p['files'] and w['before']['runtime']==p['runtime'] and w['before']['environment']==p['environment'] and w['before']['fixture']=={name:row['sha256'] for name,row in p['fixture_files'].items()}
        assert w['before']['checkpoints']=={'config':p['checkpoint_config_sha256'],'index':p['checkpoint_index_sha256']}
        assert w['application_library_loaded'] is w['gpu_model_executed'] is w['model_or_performance_qualified'] is False
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        for name in ('fp8_z128_native_control.py','gdn_z_algorithm_log.py','gdn_z_projection.py','verify_fp8_z128_native_control.py'):
            assert z.read(METHODS+name)==Path(__file__).with_name(name).read_bytes() if METHODS+name in manifest['files'] else sha(Path(__file__).with_name(name).read_bytes())==p['files'][METHODS+name]
        proof=z.read('results/'+PARENT+'-verified.json');q=json.loads(proof);assert proof==(args.proofs/(PARENT+'-verified.json')).read_bytes() and sha(proof)==p['parent']['proof_sha256']
        assert q['status']=='real_gdn_z_default_and_m32_shared_algorithm_fully_replayed' and q['archive_sha256']==p['parent']['archive_sha256']==manifest['linked_parent_archive']['archive_sha256']==sha((args.proofs/(PARENT+'-evidence.zip')).read_bytes())
        parent,unused=checked(args.proofs/(PARENT+'-evidence.zip'))
        with parent:
            fixture_raw=z.read('results/actual-gdn-z-128-fixture/manifest.json');assert fixture_raw==parent.read('results/actual-gdn-z-128-fixture/manifest.json') and sha(fixture_raw)==p['fixture_manifest_sha256']==w['fixture_manifest_sha256'];f=json.loads(fixture_raw);assert f['files']==p['fixture_files']
            for name,row in p['fixture_files'].items():
                if name.endswith('.e4m3'):continue
                raw=z.read('results/actual-gdn-z-128-fixture/'+name);assert len(raw)==row['bytes'] and sha(raw)==row['sha256'] and raw==parent.read('results/actual-gdn-z-128-fixture/'+name)
            origin=json.loads(z.read(METHODS+'evidence/fp8-z128-native-control-origin.json'));assert origin['candidate_copies']==p['candidate_copies'] and len(p['candidate_copies'])==10
            derivation=json.loads(z.read(METHODS+'evidence/fp8-z128-native-source-derivation.json'));assert derivation['qualified_control_proof_sha256']==sha(proof)
            assert derivation['opaque_configuration']==p['expected_m32_opaque']
            for name,row in p['candidate_copies'].items():assert sha(z.read(METHODS+name))==row['sha256']
            for row in derivation['files']:
                raw=z.read(METHODS+'native_fp8_z128_candidate/'+Path(row['source']).name);assert sha(raw)==row['source_sha256'];text=raw.decode()
                for step in row['steps']:assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
                assert text.encode()==z.read(METHODS+'native_fp8_z128_candidate/'+Path(row['target']).name) and sha(text.encode())==row['target_sha256']
            passed=w['exit_code']==0;profiles=[];words=0;phases=['cpu-ledger','wire-cpu-ledger','build-ledger']
            for mode in ('control','native'):
                for layer in (0,1):
                    label=mode+'-layer'+str(layer)
                    if label+'_replay' not in w:continue
                    actual=base+label+'/';result=replay_arrays(lambda name:z.read('results/actual-gdn-z-128-fixture/layer'+str(layer)+'/'+name),lambda name:z.read(actual+name),mode);log=replay_algorithms(z.read(base+label+'-ledger/commands.log'),mode,p['expected_m32_opaque'])
                    assert result==w[label+'_replay'] and log==w[label+'_algorithms'] and w[label+'_exit_code']==0
                    reference='results/'+PARENT+'/'+('shared' if mode=='native' else 'control')+'-layer'+str(layer)+'/'
                    for name in ('cast-m128.bf16','quantized-m128.e4m3','four-m32-r0.bf16','four-m32-r1.bf16','m128-r0.bf16','m128-r1.bf16'):assert z.read(actual+name)==parent.read(reference+name)
                    profiles.append({'mode':mode,'layer':layer,'replay':result,'algorithms':log});words+=result['words']['complete_bf16_output_words_replayed'];phases.append(label+'-ledger')
            wire=w.get('wire_contract')
            if passed:
                assert w['status']=='fp8_z128_actual_typed_helper_collected_needs_replay' and len(profiles)==4
                assert wire==json.loads(z.read(base+'wire-cpu/report.json')) and wire['status']=='fp8_z128_plugin_cpu_contract_passed'
                for key,value in {'wire_bytes':20,'wire_policy':0x5a313238,'bad_wires':34,'bad_constructors':18,'bad_descriptors':23,'bad_formats':11,'policy_rejections':8,'cross_wire_rejections':5,'workspace_bytes':4849664,'unexpected_gpu_calls':0,'runtime_version':101601,'header_version':101601}.items():assert wire[key]==value
                assert wire['enqueue_tested'] is wire['model_qualified'] is wire['performance_admitted'] is False
                assert PureWindowsPath(wire['loaded_tensorrt_dll'])==PureWindowsPath('D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll')
                assert b'Ran 13 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
                assert sha(z.read(base+'build/control.exe'))==w['executable_sha256']
                for phase in phases:
                    events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
                    assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files']
        result={'status':'fp8_z128_actual_typed_helper_and_plugin_fully_replayed' if passed else 'fp8_z128_actual_typed_helper_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'complete_profiles_replayed':len(profiles),'complete_bf16_output_words_replayed':words,'profiles':profiles,'wire_contract':wire,'source_runtime_sdk_fixture_unchanged':True,'candidate_copies':p['candidate_copies'],'application_library_loaded':False,'gpu_model_executed':False,'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'report_error':w.get('error'),'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'complete_output_words':words,'profiles':len(profiles)}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
