"""Complete independent native Z matrix word/log/provenance replay on either PC."""
from pathlib import Path, PureWindowsPath
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_z_projection import replay
from test2026.cuda.qwen38_5090.gdn_z_algorithm_log import replay_log
JOB='gdn-z-native-control-attempt1';EXTRACT='gdn-z-fixture-extraction-attempt1';PARENT='gdn-chunk-stage-witness-attempt2';METHODS='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def checked(path):
    z=zipfile.ZipFile(path);m=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for name,h in m['files'].items():assert sha(z.read(name))==h
    return z,m

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive)
    with z:
        base='results/'+JOB+'/';w=json.loads(z.read(base+'workflow.json'));praw=z.read('results/gdn-z-native-control-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] and w['before']==w['after']
        assert w['before']['source']==p['files'] and w['before']['runtime']==p['runtime'] and w['before']['environment']==p['environment'] and w['before']['fixture']=={name:row['sha256'] for name,row in p['fixture_files'].items()}
        assert w['before']['checkpoints']=={'config':p['checkpoint_config_sha256'],'index':p['checkpoint_index_sha256']}
        assert w['application_library_loaded'] is w['gpu_model_executed'] is w['model_or_performance_qualified'] is False
        for name in ('gdn_z_projection.py','gdn_z_algorithm_log.py','verify_gdn_z_projection.py'):
            assert z.read(METHODS+name)==Path(__file__).with_name(name).read_bytes()
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]
            assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']
        fixture_raw=z.read('results/actual-gdn-z-128-fixture/manifest.json');f=json.loads(fixture_raw);assert sha(fixture_raw)==p['fixture_manifest_sha256']==w['fixture_manifest_sha256'] and f['files']==p['fixture_files']
        with zipfile.ZipFile(args.proofs/(EXTRACT+'-evidence.zip')) as extracted:assert extracted.read('results/actual-gdn-z-128-fixture/manifest.json')==fixture_raw
        expected_external={name:row for name,row in p['fixture_files'].items() if name.endswith('.e4m3')};assert len(expected_external)==2 and manifest['external_weight_files']==expected_external
        for name,row in p['fixture_files'].items():
            if name in expected_external:continue
            raw=z.read('results/actual-gdn-z-128-fixture/'+name);assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
        passed=w['exit_code']==0;profiles=[];logs={};complete_words=0
        phases=['cpu-ledger','build-ledger']
        for mode in ('control','shared'):
            for layer in (0,1):
                key=mode+'_layer'+str(layer)
                if key+'_replay' not in w:continue
                label=mode+'-layer'+str(layer);native=base+label+'/'
                result=replay(lambda name:z.read('results/actual-gdn-z-128-fixture/layer'+str(layer)+'/'+name),lambda name:z.read(native+name),mode)
                algorithms=replay_log(z.read(base+label+'-ledger/commands.log'),mode)
                assert result==w[key+'_replay'] and algorithms==w[key+'_algorithm_replay'] and w[key+'_exit_code']==0
                profiles.append({'layer':layer,'replay':result,'algorithm_replay':algorithms});complete_words+=result['complete_bf16_output_words_replayed'];logs[(mode,layer)]=algorithms;phases.append(label+'-ledger')
        if passed:
            assert w['status']=='real_gdn_z_native_control_collected_needs_replay' and len(profiles)==4 and w['native_projection_control_executed'] is True
            assert len([name for name in z.namelist() if name.startswith(base) and name.endswith('.bf16')])==20
            assert b'Ran 8 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            assert sha(z.read(base+'build/control.exe'))==w['executable_sha256']
            for layer in (0,1):
                default=logs[('control',layer)]['calls'];shared=logs[('shared',layer)]['calls']
                assert [row['original_opaque'] for row in default]==[row['original_opaque'] for row in shared]
                assert [row['original_algorithm_id'] for row in default]==[row['original_algorithm_id'] for row in shared]
            for phase in phases:
                events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
                assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['files']
                if phase not in ('cpu-ledger','build-ledger'):
                    artifacts=events[0]['artifact_sha256'];assert artifacts[str(PureWindowsPath('D:/CantorAI/5090-study-20261023/results/'+JOB+'/build/control.exe'))]==w['executable_sha256']
        result={'status':'real_gdn_z_default_and_m32_shared_algorithm_fully_replayed' if passed else 'real_gdn_z_native_control_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'complete_bf16_output_words_replayed':complete_words,'complete_profiles_replayed':len(profiles),'profiles':profiles,'source_runtime_sdk_and_fixture_unchanged':True,'actual_rawcheckpoint_fixture_reread_before_run':True,'external_weight_files':expected_external,'application_library_loaded':False,'gpu_model_executed':False,'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'report_error':w.get('error'),'scope':p['scope']}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'complete_output_words':complete_words,'profiles':len(profiles)}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
