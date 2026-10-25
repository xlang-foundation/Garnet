"""Replay every native paged read from retained raw rows and an independent oracle."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.paged_prompt_component import case_specs,embedding_oracle,replay_case
JOB='paged-prompt-native-component-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();sha=lambda raw:hashlib.sha256(raw).hexdigest()
    archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        raw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/paged-prompt-native-component-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['cleanup_finished'] is True
        for n,h in p['new_files'].items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files']['Garnet/test2026/cuda/qwen38_5090/'+Path(__file__).name]
        parent='bf16-scores-tiled-api-model-admission-attempt1';proof_raw=z.read('results/'+parent+'-verified.json');q=json.loads(proof_raw)
        assert sha(proof_raw)==w['closed_native_proof_sha256']==p['closed_native_proof_sha256'] and q['status'] in ('native_tiled_scores_model_replayed','native_tiled_scores_model_failure_preserved')
        parent_archive=args.proofs/(parent+'-evidence.zip');assert sha(parent_archive.read_bytes())==w['closed_native_archive_sha256']==p['closed_native_archive_sha256']==q['archive_sha256']==manifest['linked_native_archive_sha256']
        with zipfile.ZipFile(parent_archive) as prior:
            old=json.loads(prior.read('results/'+parent+'/workflow.json'));assert old['cleanup_finished'] and old['baseline_library_restored'] and old['source_sha256']==old['source_after_sha256'] and old['runtime_dependencies']==old['runtime_dependencies_after']==p['runtime']
            assert p['files']=={**old['source_after_sha256'],**p['new_files']}
        passed=w['exit_code']==0;replayed=[]
        if passed:
            assert w['status']=='paged_prompt_native_component_collected_needs_replay'
            for name in ('cpu-ledger','embedded-cpu-ledger','fixture-cpu-ledger','gpu-ledger'):
                events=[json.loads(line) for line in z.read(base+name+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[-1]['event']=='finished' and type(events[-1]['exit_code']) is int and events[-1]['exit_code']==0
            log=z.read(base+'cpu-ledger/commands.log');assert b'Ran 12 tests' in log and b'\nOK' in log
            embedded=doc(base+'embedded-cpu/report.json');assert embedded=={'status':'paged_prompt_embedded_cpu_controls_passed','actual_tests':12,'gpu_executed':False}
            fixture_raw=z.read(base+'fixture/fixture.json');fixture=json.loads(fixture_raw)
            for n,h in p['retained_fixture_files'].items():assert sha(z.read(base+'fixture/'+Path(n).name))==h
            oracle=embedding_oracle(fixture,lambda n:z.read(base+'fixture/'+n))
            report=doc(base+'native/report.json');assert report['status']=='paged_prompt_native_bytes_collected_needs_replay' and report['library_sha256']==p['runtime']['garnet.dll']
            assert report['fixture_sha256']==sha(fixture_raw) and report['decoder_constructed'] is report['performance_admitted'] is report['quality_qualified'] is False
            specs=case_specs();assert [v['spec'] for v in report['cases']]==specs;observed=set()
            for record,spec in zip(report['cases'],specs):
                assert record['events_file']==spec['key']+'-events.jsonl'
                events=[json.loads(line) for line in z.read(base+'native/'+record['events_file']).splitlines()]
                replay=replay_case(record,events,lambda n:z.read(base+'native/rows/'+n),fixture,oracle,spec)
                assert replay==record['local_replay'];replayed.append(replay);observed.update(replay['raw_row_sha256'])
            assert set(report['captured_rows'])==observed and all(n==20480 for n in report['captured_rows'].values())
            actual_files={n.removeprefix(base+'native/rows/').removesuffix('.f32') for n in manifest['files'] if n.startswith(base+'native/rows/')}
            assert actual_files==observed
            resources=doc(base+'resources/resource-monitor.json');assert resources['child_exit_code']==0 and resources['gpu']['uuid']==p['gpu_uuid']
        result={'status':'paged_prompt_native_component_bytes_replayed' if passed else 'paged_prompt_native_component_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
                'library_sha256':p['runtime']['garnet.dll'],'cases':replayed,'raw_read_events':sum(v['read_events'] for v in replayed),'logical_bytes_reconstructed_and_verified':sum(v['logical_bytes_reconstructed_and_verified'] for v in replayed),
                'decoder_model_qualified':False,'serving_memory_saving_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(result));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
