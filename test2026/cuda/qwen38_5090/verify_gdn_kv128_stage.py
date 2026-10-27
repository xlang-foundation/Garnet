"""Full original GDN math replay and immutable native/runtime restoration checks."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
from test2026.cuda.qwen38_5090.gdn_kv128_stage_replay import replay
from test2026.cuda.qwen38_5090.z128_reconstructed_layer2_trace import replay as trace_replay
JOB='gdn-kv128-stage-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proofs',type=Path,required=True);a=ap.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/gdn-kv128-stage-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}}
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints'] and not w['performance_admitted']
        assert w['candidate_library_sha256']==p['candidate_library_sha256']=='5dff30dd8d2960bcfe64cc25d5014874bd893a3649407ea157eaa7759179350f' and w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==50
        for n,h in {**p['new_files'],**p['audit_dependencies'],**p['checkpoints']}.items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files'][P+Path(__file__).name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof)
            assert sha(proof)==row['proof_sha256'] and proof==(a.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]==sha((a.proofs/(job+'-evidence.zip')).read_bytes())
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==50 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert len(expected)==14 and original.testzip() is None and {n:sha(original.read(n)) for n in original.namelist()}==expected
        def command(phase,expected_exit=0):
            events=[json.loads(line) for line in z.read(base+phase+'-ledger/commands.jsonl').splitlines()]
            assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==expected_exit and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
        command('cpu');assert b'Ran 27 tests' in z.read(base+'cpu-ledger/commands.log') and b'Ran 86 tests' in z.read(base+'cpu-ledger/commands.log')
        arrays=traced=None;passed=w['exit_code']==0
        if passed:
            assert w['status']=='GDN50_stage_collected_needs_replay';command('embedded-metadata');command('native')
            m=doc(base+'embedded-metadata.json');assert m['status']=='embedded_GDN50_metadata_import_passed' and m['numpy_imported'] is m['GPU_or_Garnet_requested'] is False
            r=doc(base+'native/report.json');assert r['candidate_library_sha256']==p['candidate_library_sha256']
            case=z.read(P+'evidence/native-host-stage-cases.json');assert sha(case)==p['audit_dependencies'][P+'evidence/native-host-stage-cases.json']==r['cases_sha256'] and r['input_ids']==json.loads(case)['cases'][0]['prompt_ids'][:128]
            arrays=replay(r,lambda n:z.read(base+'native/'+n),a.proofs);assert arrays==doc(base+'array-analysis.json') and arrays['controlled_first128_all11_stages_exact']
            traced=trace_replay(z.read(base+'native-ledger/commands.log'),True);assert json.loads(json.dumps(traced))==doc(base+'trace-analysis.json') and set(traced['plugin']['routes'])=={'fp8','nvfp4','fp8_m128','fp8_z128_reconstructed'} and not traced['plugin']['failures'] and traced['factory']['factory_calls']>=4
            resources=doc(base+'native-resources/resource-monitor.json');assert resources['child_exit_code']==0 and resources['gpu']['uuid']==p['gpu_uuid']
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='actual_GDN50_all_original_stages_states_and_509_control_entries_replayed' if passed else 'actual_GDN50_stage_failure_preserved',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),candidate_library_sha256=p['candidate_library_sha256'],source_runtime_SDK_checkpoint_restored=True,arrays=arrays,trace=traced,full_cache_KEY_RoPE_six_full4K_quality_gates_still_required=True,full_model_or_quality_or_performance_qualified=False,two_times_target_achieved=False,verifier_sha256=sha(Path(__file__).read_bytes()))
    finally:z.close()
    target=a.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),arrays=arrays)))
if __name__=='__main__':main()
