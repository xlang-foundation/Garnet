"""Independent exploratory cache/API replay; never clear the failed build gate."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_gdn_chunk_stage_witness_bound import checked
from test2026.cuda.qwen38_5090.fp8_policy_cache_production_contract import CASES,replay
JOB='fp8-policy-cache-production-diagnostic-attempt2';P='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive);base='results/'+JOB+'/'
    try:
        doc=lambda n:json.loads(z.read(n));raw=z.read(P+'evidence/fp8-policy-cache-production-python-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json');passed=w['exit_code']==0
        assert sha(raw)==w['preparation_sha256'] and type(w['exit_code']) is int and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['native_build_artifact_identity_gate_passed'] is p['native_build_artifact_identity_gate_passed'] is False and w['exploratory_only'] is True and w['model_or_performance_admitted'] is False
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['CUDA_objects_before']==w['CUDA_objects_after']==p['current_cuda_objects'] and w['candidate_library_sha256']==p['candidate_library_sha256']=='dc15e4f53855fb91e182cf52e78eb3917068b8df80cde1242ff1799ce4fb99ba'
        assert len(p['candidate_files'])==62 and w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base']
        for n,h in {**p['new_files'],**p['checkpoints'],**p['audit_dependencies']}.items():assert sha(z.read(n))==h
        assert sha(Path(__file__).read_bytes())==p['new_files'][P+Path(__file__).name]
        for name,row in p['parents'].items():
            proof=z.read('results/'+name+'-verified.json');q=json.loads(proof);assert sha(proof)==row['proof_sha256'] and proof==(args.proofs/(name+'-verified.json')).read_bytes() and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][name]==sha((args.proofs/(name+'-evidence.zip')).read_bytes())
        failed=json.loads((args.proofs/'fp8-policy-cache-build-attempt1-verified.json').read_bytes());assert failed['exit_code']==1 and failed['all34_CUDA_objects_byteexact'] is False
        assert sha(z.read(p['source_bundle']))==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(z.read(p['source_bundle']))) as source:assert len(source.namelist())==len(set(source.namelist()))==62 and {n:sha(source.read(n)) for n in source.namelist()}==p['candidate_files']
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:assert {n:sha(old.read(n)) for n in old.namelist()}=={n:h for n,h in p['candidate_base'].items() if h is not None}
        def command(phase,required,expected):
            path=base+phase+'-ledger/commands.jsonl'
            if path not in manifest['files']:assert not required;return None
            events=[json.loads(line) for line in z.read(path).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==expected
            if required:assert events[1]['exit_code']==0
            return events[1]['exit_code']
        command('cpu',passed,p['files']);r=doc(base+'native/report.json') if base+'native/report.json' in manifest['files'] else None;arrays=dict(status='exploratory_cache_failure_preserved',report_status=None if r is None else r['status'],error=None if r is None else r.get('error'),captured_cases=0 if r is None else len(r['cases']))
        if 'installed_source_sha256' in w:assert w['installed_source_sha256']==p['installed_files']
        else:assert not passed
        for phase in ('selector-cpu','embedded-selector-cpu','tokenizer-cpu','gpu'):command(phase,passed,p['installed_files'])
        if base+'resources/resource-monitor.json' in manifest['files']:assert doc(base+'resources/resource-monitor.json')['gpu']['uuid']==p['gpu_uuid']
        if passed:
            assert w['status']=='actual_production_cache_diagnostic_collected_needs_replay';arrays=replay(r,p['candidate_library_sha256']);assert arrays==r['local_replay']==doc(base+'cache-analysis.json')
            for row,spec in zip(r['cases'],CASES):
                assert row==doc(base+'native/'+spec['key']+'/report.json');command('native/'+spec['key'],True,p['installed_files'])
                assert ('ACTUAL_PRODUCTION_CACHE_CASE_PASS '+spec['key']).encode() in z.read(base+'native/'+spec['key']+'-ledger/commands.log')
                for phase in ('before','after'):
                    for name,item in row['cache_'+phase].items():
                        b=z.read(base+'native/'+spec['key']+'/cache-'+phase+'/'+name);assert len(b)==item['bytes'] and sha(b)==item['sha256']
                        if name=='runtime_graph.cache':assert b.splitlines()[1].decode('ascii')==item['fingerprint']
            for phase,status in (('selector-cpu','native_tiled_selector_cpu_exact'),('embedded-selector-cpu','embedded_native_tiled_selector_exact')):
                v=doc(base+phase+'/report.json');assert v['status']==status and v['library_sha256']==p['candidate_library_sha256'] and len(v['cases'])==10 and v['gpu_executed'] is False
                for row in v['cases']:assert row['exit_code']==0 and row['actual']['first']==row['actual']['after_environment_flip']==(row['environment_value']=='1') and row['actual']['gpu_executed'] is False
            with zipfile.ZipFile(io.BytesIO(z.read('results/tokenizer-nfc-cpu-oracles.zip'))) as oracle:
                oracle_rows=[json.loads(line) for line in oracle.read('cases.jsonl').splitlines()]
            actual=[json.loads(line) for line in z.read(base+'tokenizer-cpu/cases.jsonl').splitlines()];assert len(actual)==len(oracle_rows)==955
            for found,expected in zip(actual,oracle_rows):assert found['key']==expected['key'] and found['kind']==expected['kind'] and found['code']==0 and found['actual_ids']==expected['expected_ids'] and found['exact'] is True
            report=doc(base+'tokenizer-cpu/report.json');assert report['status']=='rebuilt_native_tokenizer_cpu_exact' and report['cases']==955 and report['mismatches']==[] and report['library_sha256']==p['candidate_library_sha256']
        else:assert w['status']=='failed' and w.get('error')
        result=dict(status='actual_exploratory_production_cache_and_native_API_replayed' if passed else 'exploratory_production_cache_failure_replayed',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),exit_code=w['exit_code'],error=w.get('error'),candidate_library_sha256=p['candidate_library_sha256'],arrays=arrays,source_runtime_SDK_checkpoint_restored=True,all34_object_build_identity_gate_passed=False,exploratory_only=True,production_cache_behavior_observed=passed,native_build_or_VLM_quality_or_performance_qualified=False,two_times_target_achieved=False)
    finally:z.close()
    blob=json.dumps(result,indent=2).encode();t=args.proofs/(JOB+'-verified.json');assert not t.exists() or t.read_bytes()==blob;t.write_bytes(blob);print(json.dumps(dict(status=result['status'],proof_sha256=sha(blob),arrays=arrays)))
if __name__=='__main__':main()
