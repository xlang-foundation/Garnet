"""Verify actual native build evidence; observation build is not numerical admission."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
JOB='full-chunk-stage-build-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/full-chunk-stage-build-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json')
        assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['base_files'] and w['methods']==w['methods_after']==p['methods']
        assert w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['cache_after_sha256']==p['cache_before_sha256']==sha(z.read(base+'CMakeCache-before.txt'))
        assert w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base'] and len(p['candidate_files'])==51
        assert w['gpu_model_executed'] is w['numerical_or_performance_admitted'] is False
        for n,h in p['methods'].items():assert sha(z.read(n))==h
        for n,h in p['checkpoints'].items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['methods'][P+Path(__file__).name] and sha(z.read(p['bundle']))==p['bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['bundle']))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==51 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
            expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert len(expected)==14 and old.testzip() is None and {n:sha(old.read(n)) for n in old.namelist()}==expected
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(job+'-verified.json')).read_bytes()
            assert q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        passed=w['exit_code']==0
        def command(name,expected_sources):
            events=[json.loads(line) for line in z.read(base+name+'-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected_sources
        if passed:
            assert w['status']=='full_chunk_stage_native_built_CPU_qualified_model_unqualified'
            before={**p['base_files'],**p['methods']};installed={**before,**{'Garnet/'+n:h for n,h in p['candidate_files'].items()}};assert w['installed_source']==installed
            command('cpu-before',before);assert b'Ran 41 tests' in z.read(base+'cpu-before-ledger/commands.log') and b'Ran 86 tests' in z.read(base+'cpu-before-ledger/commands.log')
            for name in ('profile','native-build','selector-cpu','embedded-selector-cpu','tokenizer-cpu'):command(name,installed)
            assert doc(base+'profile/report.json')==dict(status='full_chunk_stage_profile_CPU_contract_passed',cases=142,selected=2,rejected=140,GPU_or_model_executed=False)
            from test2026.cuda.qwen38_5090.run_z128_factory_trace_native_build import validate_cuda_objects
            assert w['cuda_objects_before']==p['cuda_objects_before'] and validate_cuda_objects(w['cuda_objects_before'],w['cuda_objects_built'])==w['changed_cuda_objects']
            assert sha(z.read(w['candidate_saved']))==w['candidate_library_sha256']!=p['runtime']['garnet.dll']
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='full_chunk_stage_native_CPU_build_replayed' if passed else 'full_chunk_stage_native_build_failure_replayed',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),candidate_files=51,candidate_library_sha256=w.get('candidate_library_sha256'),source_runtime_SDK_checkpoint_CMake_restored=True,current_CPU_controls=41 if passed else None,historical_controls=86 if passed else None,actual_header_CPU_cases=142 if passed else None,GPU_model_or_matmul_executed=False,model_or_quality_or_performance_qualified=False,two_times_target_achieved=False)
    finally:z.close()
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(result))
if __name__=='__main__':main()
