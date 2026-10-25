"""Independently replay the actual candidate build and CPU-only native gates."""
from pathlib import Path
import argparse,hashlib,json,zipfile

JOB='bf16-scores-tiled-native-build-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n))
        manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        w=doc(base+'workflow.json');praw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/bf16-scores-tiled-native-build-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] is True
        assert w['source_before']==w['source_after']==p['base_files']
        assert w['runtime_before']==w['runtime_after']==p['runtime']
        assert w['environment_before']==w['environment_after']==p['environment']
        assert w['methods_after']==p['methods']
        assert w['baseline_source_restored'] is w['baseline_library_restored'] is True
        assert sha(z.read(base+'original-overlay.cu'))==p['base_files']['Garnet/src/cuda/qwen38_bf16_kv_low_batch_overlay.cu']
        assert sha(z.read(base+'candidate-overlay.cu'))==p['candidate_source_sha256']
        assert sha(z.read(base+'CMakeCache-before.txt'))==p['cache_before_sha256']
        for n,h in p['methods'].items():assert sha(z.read(n))==h
        parent_raw=z.read('results/native-cuda-phase-logged-bootstrap-attempt1-verified.json');parent=json.loads(parent_raw)
        assert sha(parent_raw)==w['bootstrap_proof_sha256']==p['bootstrap_proof_sha256'] and parent['status']=='native_cuda_logged_bootstrap_replayed'
        component=doc('results/bf16-scores-tiled-correctness-attempt1-verified.json')
        assert component['status']=='bf16_scores_tiled_fp64_replayed' and component['archive_sha256']==w['component_archive_sha256']==p['component_archive_sha256']
        assert type(w['exit_code']) is int;passed=w['exit_code']==0
        if passed:
            assert w['status']=='native_tiled_scores_built_cpu_qualified_model_unqualified'
            for ledger in ('cpu-ledger','build-ledger','selector-cpu-ledger','tokenizer-cpu-ledger'):
                events=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[-1]['event']=='finished' and type(events[-1]['exit_code']) is int and events[-1]['exit_code']==0
            log=z.read(base+'cpu-ledger/commands.log');assert b'Ran 49 tests' in log and b'\nOK' in log and b'FAILED' not in log
            assert set(w['cuda_objects_before'])==set(w['cuda_objects_built'])
            changed=[n for n,h in w['cuda_objects_before'].items() if w['cuda_objects_built'][n]!=h]
            assert changed==w['changed_cuda_objects'] and len(changed)==1 and changed[0].endswith('/qwen38_bf16_kv_low_batch_overlay.cu.obj')
            assert sha(z.read(w['candidate_saved']))==w['candidate_library_sha256']!=p['runtime']['garnet.dll']
            selector=doc(base+'selector-cpu/report.json');assert selector['status']=='native_tiled_selector_cpu_exact' and selector['library_sha256']==w['candidate_library_sha256'] and selector['gpu_executed'] is False
            expected=(None,'','0','1','true','01',' 1','1 ','2','-1');assert len(selector['cases'])==len(expected)
            for i,(row,value) in enumerate(zip(selector['cases'],expected)):
                assert row['environment_value']==value and type(row['exit_code']) is int and row['exit_code']==0
                actual=json.loads(z.read(base+'selector-cpu/'+str(i)+'-stdout.txt'))
                assert actual==row['actual'] and actual['first']==actual['after_environment_flip']==int(value=='1') and actual['gpu_executed'] is False
            tokenizer=doc(base+'tokenizer-cpu/report.json');assert tokenizer['status']=='rebuilt_native_tokenizer_cpu_exact' and tokenizer['library_sha256']==w['candidate_library_sha256'] and tokenizer['cases']==955 and tokenizer['mismatches']==[] and tokenizer['gpu_model_executed'] is False
            raw=z.read(base+'tokenizer-cpu/cases.jsonl');assert sha(raw)==tokenizer['raw_cases_sha256']
            rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==955 and all(v['exact'] is True and v['code']==0 for v in rows)
        result={'status':'native_tiled_scores_build_cpu_replayed' if passed else 'native_tiled_scores_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
                'candidate_library_sha256':w.get('candidate_library_sha256'),'baseline_library_sha256':p['runtime']['garnet.dll'],'source_and_runtime_restored':True,
                'cuda_objects_changed':w.get('changed_cuda_objects'),'gpu_model_executed':False,'performance_admitted':False,'two_times_target_achieved':False,
                'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode()
    assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps(result));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
