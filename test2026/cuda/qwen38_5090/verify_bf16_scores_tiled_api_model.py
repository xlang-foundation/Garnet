"""Two new-DLL routes versus retained raw model oracle; no timing admission."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.tiled_scores_model import replay_tiled_model

JOB='bf16-scores-tiled-api-model-admission-attempt1';PARENT='mtp-prefill16-full4k-admission-attempt1';BUILD='bf16-scores-tiled-native-api-build-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    sha=lambda b:hashlib.sha256(b).hexdigest()
    code=subprocess.run([sys.executable,str(Path(__file__).with_name('verify_bf16_scores_tiled_native_api_build.py')),'--proofs',str(args.proofs)]).returncode
    assert code==0
    build_proof_raw=(args.proofs/(BUILD+'-verified.json')).read_bytes();build_proof=json.loads(build_proof_raw)
    baseline=args.proofs/(PARENT+'-evidence.zip');archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z,zipfile.ZipFile(baseline) as old,zipfile.ZipFile(args.proofs/(BUILD+'-evidence.zip')) as built:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        p_raw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/bf16-scores-tiled-api-model-preparation.json');p=json.loads(p_raw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(p_raw)==w['preparation_sha256'] and w['cleanup_finished'] is w['baseline_library_restored'] is True
        assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime']
        assert w['environment_before']==w['environment_after']==p['environment']
        assert w['candidate_library_sha256']==p['candidate_library_sha256']==build_proof['candidate_library_sha256']
        assert w['baseline_library_sha256']==p['runtime']['garnet.dll']==build_proof['baseline_library_sha256']
        assert sha(build_proof_raw)==w['build_proof_sha256']==p['build_proof_sha256'] and z.read('results/'+BUILD+'-verified.json')==build_proof_raw
        assert sha(baseline.read_bytes())==w['baseline_archive_sha256']==p['baseline_archive_sha256']==manifest['linked_parent_archive_sha256']
        assert build_proof['archive_sha256']==manifest['linked_build_archive_sha256']
        original_manifest=json.loads(old.read('evidence-manifest.json'));assert old.testzip() is None
        for n,h in original_manifest['files'].items():assert sha(old.read(n))==h
        original=json.loads(old.read('results/'+PARENT+'/workflow.json'));assert original['exit_code']==0 and original['cleanup_finished'] and original['library_unchanged']
        reference=json.loads(old.read('results/'+PARENT+'/native/report.json'))
        cases=json.loads(old.read('Garnet/test2026/cuda/qwen38_5090/evidence/native-host-stage-cases.json'))['cases']
        prior=json.loads(built.read('Garnet/test2026/cuda/qwen38_5090/evidence/bf16-scores-tiled-native-api-build-preparation.json'))
        inherited={**prior['base_files'],**prior['methods'],'Garnet/test2026/cuda/qwen38_5090/verify_bf16_scores_tiled_native_api_build.py':build_proof['verifier_sha256']}
        assert p['files']=={**inherited,**p['new_files']}
        for n,h in p['new_files'].items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files']['Garnet/test2026/cuda/qwen38_5090/'+Path(__file__).name]
        passed=w['exit_code']==0;replayed={}
        if passed:
            assert w['status']=='native_tiled_scores_model_collected_needs_replay' and set(w['modes'])=={'default','tiled'}
            for ledger in ('cpu-ledger','default-ledger','tiled-ledger'):
                events=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[-1]['event']=='finished' and events[-1]['exit_code']==0
            log=z.read(base+'cpu-ledger/commands.log');assert b'Ran 10 tests' in log and b'\nOK' in log
            for mode in ('default','tiled'):
                assert type(w['modes'][mode]['exit_code']) is int and w['modes'][mode]['exit_code']==0
                report=json.loads(z.read(base+mode+'/report.json'))
                assert report['library_sha256']==p['candidate_library_sha256'] and report['tiled_scores_enabled']==(mode=='tiled') and report['baseline_full4k_archive_sha256']==p['baseline_archive_sha256']
                actual=replay_tiled_model(report,lambda n:z.read(base+mode+'/'+n),reference,lambda n:old.read('results/'+PARENT+'/native/'+n),cases)
                assert actual==report['local_replay'];replayed[mode]=actual
                resources=json.loads(z.read(base+mode+'-resources/resource-monitor.json'))
                assert type(resources['child_exit_code']) is int and resources['child_exit_code']==0
        result={'status':'native_tiled_scores_model_replayed' if passed else 'native_tiled_scores_model_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
                'candidate_library_sha256':p['candidate_library_sha256'],'baseline_library_sha256':p['runtime']['garnet.dll'],'baseline_archive_sha256':p['baseline_archive_sha256'],
                'routes':replayed,'baseline_source_and_runtime_restored':True,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(result));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
