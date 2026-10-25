"""Replay actual native and embedded API compatibility before model retry."""
from pathlib import Path
import argparse,hashlib,io,json,zipfile
JOB='bf16-scores-tiled-native-api-build-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();sha=lambda raw:hashlib.sha256(raw).hexdigest()
    archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        p_raw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/bf16-scores-tiled-native-api-build-preparation.json');p=json.loads(p_raw);w=doc(base+'workflow.json')
        assert sha(p_raw)==w['preparation_sha256'] and w['cleanup_finished'] is w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert w['source_before']==w['source_after']==p['base_files'] and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment']
        assert w['methods_after']==p['methods']
        assert sha(z.read(base+'CMakeCache-before.txt'))==p['cache_before_sha256']
        for label in ('original','candidate'):
            with zipfile.ZipFile(io.BytesIO(z.read(base+label+'-source.zip'))) as s:
                assert s.testzip() is None and set(s.namelist())==set(p['candidate_files'])
                for n in s.namelist():assert sha(s.read(n))==(p['candidate_files'][n] if label=='candidate' else p['base_files']['Garnet/'+n])
        for n,h in p['methods'].items():assert sha(z.read(n))==h
        failure_raw=z.read('results/bf16-scores-tiled-model-admission-attempt1-verified.json');failure=json.loads(failure_raw)
        assert sha(failure_raw)==w['failed_model_proof_sha256']==p['failed_model_proof_sha256'] and failure['status']=='native_tiled_scores_model_failure_preserved'
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='native_tiled_scores_api_built_cpu_qualified_model_unqualified'
            for name in ('cpu','build','selector-cpu','embedded-selector-cpu','tokenizer-cpu'):
                events=[json.loads(line) for line in z.read(base+name+'-ledger/commands.jsonl').splitlines()]
                assert len(events)==2 and events[-1]['event']=='finished' and type(events[-1]['exit_code']) is int and events[-1]['exit_code']==0
            log=z.read(base+'cpu-ledger/commands.log');assert b'Ran 49 tests' in log and b'\nOK' in log
            assert sha(z.read(w['candidate_saved']))==w['candidate_library_sha256']!=p['runtime']['garnet.dll']
            assert set(w['cuda_objects_before'])==set(w['cuda_objects_built'])
            changed=[n for n,h in w['cuda_objects_before'].items() if w['cuda_objects_built'][n]!=h];assert changed==w['changed_cuda_objects']
            assert all(n.endswith('/qwen38_bf16_kv_low_batch_overlay.cu.obj') for n in changed)
            for name,status in (('selector-cpu','native_tiled_selector_cpu_exact'),('embedded-selector-cpu','embedded_native_tiled_selector_exact')):
                report=doc(base+name+'/report.json');assert report['status']==status and report['library_sha256']==w['candidate_library_sha256'] and report['gpu_executed'] is False
                values=(None,'','0','1','true','01',' 1','1 ','2','-1');assert len(report['cases'])==len(values)
                for i,(row,value) in enumerate(zip(report['cases'],values)):
                    actual=doc(base+name+'/'+str(i)+'-stdout.txt');assert row['actual']==actual and row['environment_value']==value and row['exit_code']==0
                    assert actual['first']==actual['after_environment_flip']==(value=='1') and actual['gpu_executed'] is False
                    if name=='embedded-selector-cpu':assert type(actual['first']) is bool
            tokenizer=doc(base+'tokenizer-cpu/report.json');assert tokenizer['status']=='rebuilt_native_tokenizer_cpu_exact' and tokenizer['library_sha256']==w['candidate_library_sha256'] and tokenizer['cases']==955 and tokenizer['mismatches']==[]
            raw=z.read(base+'tokenizer-cpu/cases.jsonl');assert sha(raw)==tokenizer['raw_cases_sha256'];rows=[json.loads(line) for line in raw.splitlines()]
            assert len(rows)==955 and all(r['exact'] is True and r['code']==0 for r in rows)
        result={'status':'native_tiled_scores_api_build_cpu_replayed' if passed else 'native_tiled_scores_api_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
                'candidate_library_sha256':w.get('candidate_library_sha256'),'baseline_library_sha256':p['runtime']['garnet.dll'],'source_and_runtime_restored':True,
                'embedded_native_api_qualified':passed,'gpu_model_executed':False,'performance_admitted':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(result));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
