"""Replay complete ordinary/canonical-cache evidence; preserve every failed gate."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
from test2026.cuda.qwen38_5090.kv128_prefix_cache_replay import replay
JOB='kv128-prefix-cache-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/kv128-prefix-cache-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}}
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==50 and w['candidate_library_sha256']==p['candidate_library_sha256']=='5dff30dd8d2960bcfe64cc25d5014874bd893a3649407ea157eaa7759179350f'
        for n,h in {**p['new_files'],**p['audit_dependencies'],**p['checkpoints']}.items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files'][P+Path(__file__).name]
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof)
            assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(job+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][job]==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==50 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
            expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert len(expected)==14 and original.testzip() is None and {n:sha(original.read(n)) for n in original.namelist()}==expected
        def command(phase,required):
            name=base+phase+'-ledger/commands.jsonl'
            if name not in manifest['files']:assert not required;return None
            events=[json.loads(line) for line in z.read(name).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
            if required:assert events[1]['exit_code']==0
            return events[1]['exit_code']
        cpu_exit=command('cpu',False);assert type(cpu_exit) is int
        if cpu_exit==0:
            log=z.read(base+'cpu-ledger/commands.log');assert b'Ran 35 tests' in log and b'Ran 86 tests' in log
        else:assert w['exit_code']!=0
        passed=w['exit_code']==0
        if base+'embedded-metadata.json' in manifest['files']:
            m=doc(base+'embedded-metadata.json');assert m['status']=='embedded_KV128_prefix_metadata_passed' and m['numpy_imported'] is m['GPU_or_Garnet_requested'] is False;command('embedded-metadata',True)
        reports={};reads={}
        for phase in ('baseline','native'):
            exit_code=command(phase,passed);name=base+phase+'/report.json'
            if name in manifest['files']:
                r=doc(name);reports[phase]=r;reads[phase]=lambda n,phase=phase:z.read(base+phase+'/'+n)
                assert r['candidate_library_sha256']==p['candidate_library_sha256']
            resource=base+phase+'-resources/resource-monitor.json'
            if resource in manifest['files']:
                m=doc(resource);assert m['gpu']['uuid']==p['gpu_uuid'] and m['child_exit_code']==exit_code
        arrays=replay(reports,reads,args.proofs,passed)
        if passed:
            assert w['status']=='KV128_prefix_cache_and_ordinary_control_collected_needs_replay' and arrays['cache_comparison']==doc(base+'cache-analysis.json')
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='actual_KV128_full_chain_prefix_caches_and_strict_ordinary_T32_oracle_replayed' if passed else 'actual_KV128_full_chain_prefix_or_ordinary_oracle_failure_preserved',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),candidate_library_sha256=p['candidate_library_sha256'],source_library_runtime_SDK_checkpoint_restored=True,arrays=arrays,full_six_case_MTP_quality_and_performance_still_required=True,model_quality_or_performance_qualified=False,two_times_target_achieved=False,verifier_sha256=sha(Path(__file__).read_bytes()))
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],proof_sha256=sha(raw),arrays=arrays)))
if __name__=='__main__':main()
