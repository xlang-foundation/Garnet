"""Replay complete diagnostic arrays and actual native plugin routes; never infer full-model parity."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fp8_z128_full4k_model import checked
from test2026.cuda.qwen38_5090.kv128_projection_replay import replay_control as replay,factory_replay
from test2026.cuda.qwen38_5090.kv128_plugin_init_contract import events,replay as replay_init
JOB='kv128-projection-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/kv128-projection-gpu-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json');assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int
        assert w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['installed_source_sha256']==p['installed_files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        for n,h in p['new_files'].items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files'][P+Path(__file__).name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        cpu=[json.loads(line) for line in z.read(base+'cpu-ledger/commands.jsonl').splitlines()];assert len(cpu)==2 and cpu[1]['exit_code']==0 and cpu[0]['source']['owned_source_sha256']==cpu[1]['source_after']['owned_source_sha256']==p['installed_files'];log=z.read(base+'cpu-ledger/commands.log');assert b'Ran '+str(p['expected_cpu_test_count']).encode()+b' tests' in log and b'Ran 86 tests' in log
        passed=w['exit_code']==0;arrays=trace=None
        if passed:
            assert w['status']=='FA3_projection_witness_collected_needs_replay';r=doc(base+'native/report.json');assert r['candidate_library_sha256']==p['candidate_library_sha256'];fixture=z.read(P+'evidence/reconstructed-fa3-common-input.f32');assert sha(fixture)==p['common_input_sha256']
            metadata=doc(base+'embedded-metadata.json');assert metadata['status']=='embedded_FA3_metadata_import_passed' and metadata['numpy_imported'] is metadata['GPU_or_Garnet_requested'] is False
            em=[json.loads(line) for line in z.read(base+'embedded-metadata-ledger/commands.jsonl').splitlines()];assert len(em)==2 and em[1]['exit_code']==0 and em[0]['source']['owned_source_sha256']==em[1]['source_after']['owned_source_sha256']==p['installed_files']
            arrays=replay(r,lambda n:z.read(base+'native/'+n),fixture,args.proofs);assert arrays==doc(base+'array-analysis.json')
            ledger=[json.loads(line) for line in z.read(base+'native-ledger/commands.jsonl').splitlines()];assert len(ledger)==2 and ledger[1]['exit_code']==0 and ledger[0]['source']['owned_source_sha256']==ledger[1]['source_after']['owned_source_sha256']==p['installed_files']
            resource=doc(base+'native-resources/resource-monitor.json');assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==0
            rows=events(z.read(base+'native-ledger/commands.log'));trace=replay_init(rows);assert trace==doc(base+'trace-analysis.json') and set(trace['routes'])=={'fp8','fp8_kv128'} and all(trace['routes'][n]>=2 for n in ('fp8','fp8_kv128')) and trace['failures']==[] and all((r['rows'],r['columns'],r['policy'])==(1024,5120,0x4b313238 if r['route']=='fp8_kv128' else 0) for r in rows)
            factory=factory_replay(z.read(base+'native-ledger/commands.log'),True);assert factory==doc(base+'factory-analysis.json') and arrays['original_eight_K_and_eight_V_differences_removed']
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='actual_KV128_projection_exact_intervention_replayed' if passed else 'actual_KV128_projection_failure_preserved',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),candidate_library_sha256=p['candidate_library_sha256'],source_runtime_SDK_checkpoint_restored=True,arrays=arrays,actual_native_FP8_plugin_initialization=trace,full_GDN_rerun_and_full4K_quality_gates_still_required=True,full_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,verifier_sha256=sha(Path(__file__).read_bytes()))
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),arrays=arrays)));return 0
if __name__=='__main__':raise SystemExit(main())
