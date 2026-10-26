"""Replay full actual canonical states with independent CPU integer BF16 oracle."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.bf16_prefill128_commit_oracle import decode,fixture_words,expected_state,GUARD
from test2026.cuda.qwen38_5090.bf16_prefill128_layout import build_bf16_prefill128_layout
from test2026.cuda.qwen38_5090.run_bf16_prefill128_commit_component import JOB,REFERENCE
sha=lambda b:hashlib.sha256(b).hexdigest()

def inventory():
    rows=[(s,c,False,1,128,0) for s,c in ((0,0),(0,1),(0,127),(0,128),(1,128),(511,128),(512,128),(513,128),(2047,2),(3968,128),(3969,127),(4095,1),(4096,0))]
    rows += [(511,128,True,1,128,0)]*3
    rows += [(0,128,False,1,128,d) for d in (1,2,3,4,5,6,10,11,12,13,14,15,16,20,21,22,23,24,25,26)]
    rows += [(0,128,False,b,128,7) for b in (0,2,4,8)]
    rows += [(0,128,False,1,c,8) for c in (0,4,32,64,256)]
    rows += [(s,c,False,1,128,9) for s,c in ((-1,1),(4097,0),(0,-1),(0,129),(3969,128),(4096,1))]
    return rows

def main():
    import numpy as np
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/';prefix='Garnet/test2026/cuda/qwen38_5090/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        w=doc(base+'workflow.json');praw=z.read(prefix+'evidence/prefill128-commit-component-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] is True and type(w['exit_code']) is int
        assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
        assert w['environment']==w['environment_after']==p['environment'] and w['sdk_before']==w['sdk_after']
        for n,h in p['files'].items():assert sha(z.read(n))==h
        profile=p['profiled_parent'];profile_raw=z.read('results/'+profile['job']+'-verified.json');profile_proof=json.loads(profile_raw)
        assert sha(profile_raw)==profile['proof_sha256'] and profile_proof['status']=='native_cuda_stdio_profiled_replayed'
        assert sha((args.proofs/(profile['job']+'-evidence.zip')).read_bytes())==profile_proof['archive_sha256']==profile['archive_sha256']
        for n in ('bf16_prefill128_commit_oracle.py','bf16_prefill128_layout.py','verify_bf16_prefill128_commit_component.py'):
            assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][prefix+n]
        parent=w['closed_reference'];assert parent['job']==REFERENCE and sha((args.proofs/(REFERENCE+'-evidence.zip')).read_bytes())==parent['archive_sha256']
        with zipfile.ZipFile(args.proofs/(REFERENCE+'-evidence.zip')) as previous:
            closed=json.loads(previous.read('results/'+REFERENCE+'/workflow.json'));assert closed==doc('results/'+REFERENCE+'/workflow.json')
            assert type(closed['exit_code']) is int and closed['exit_code']==parent['exit_code'] and closed['cleanup_finished'] is True
            assert closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
        passed=w['exit_code']==0;replayed=None
        if passed:
            assert w['status']=='prefill128_commit_component_collected_needs_replay'
            for ledger in ('cpu-ledger','build-ledger','gpu-ledger'):
                records=[json.loads(v) for v in z.read(base+ledger+'/commands.jsonl').splitlines()]
                assert len(records)==2 and type(records[-1]['exit_code']) is int and records[-1]['exit_code']==0
            assert b'Ran 10 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            assert sha(z.read('tools/bf16-kv-native/prefill128-commit-probe.exe'))==w['executable_sha256']
            layout=doc(prefix+'evidence/prefill128-commit-probe-layout.json');assert layout==build_bf16_prefill128_layout()
            target=decode(z.read(base+'native/target-input.rle'),44564480+64)
            assert np.all(target[:32]==GUARD) and np.all(target[-32:]==GUARD)
            assert np.array_equal(target[32:-32],fixture_words(44564480))
            final=decode(z.read(base+'native/target-final.rle'),44564480+64);assert np.array_equal(target,final);del final
            rows=[json.loads(v) for v in z.read(base+'native/cases.jsonl').splitlines()];expected=inventory();assert len(rows)==len(expected)==51
            words=0
            for i,(row,case) in enumerate(zip(rows,expected)):
                start,count,graph,batch,chunk,defect=case
                assert all(type(row[k]) is int for k in ('id','start','count','batch','chunk','defect','graph_nodes','full_state_words_compared','target_words_compared'))
                assert row['graph'] is graph
                assert row['id']==i and (row['start'],row['count'],row['graph'],row['batch'],row['chunk'],row['defect'])==case
                assert type(row['return_code']) is int and (row['return_code']==0)==(defect==0)
                assert row['graph_nodes']==(2 if graph else 0) and row['target_readonly'] is row['full_state_exact'] is True
                assert row['canonical_file']=='case-'+str(i)+'.rle' and row['full_state_words_compared']==106823680+64 and row['target_words_compared']==44564480+64
                actual=decode(z.read(base+'native/'+row['canonical_file']),106823680+64)
                oracle=expected_state(layout,start if defect==0 else 0,count if defect==0 else 0,target[32:-32])
                assert np.array_equal(actual,oracle),('Independent full canonical replay failed',i);words+=len(actual);del actual,oracle
            resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(v) for v in z.read(base+'resources/resource-vram.jsonl').splitlines()]
            assert type(resource['child_exit_code']) is int and resource['child_exit_code']==0 and resource['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
            gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])];assert samples and all(v>0 for v in gaps)
            memory=resource['sampled_memory'];assert memory['samples']==len(samples) and memory['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in samples)
            assert memory['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None) and memory['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None)
            replayed={'valid_cases':16,'invalid_cases':35,'graph_replays':3,'full_canonical_words_replayed':words,
                      'full_target_input_and_final_replayed':True,'target_readonly_checks_reported':51,'component_correctness_sampled_memory':memory,'runtime_or_model_qualified':False}
        result={'status':'prefill128_commit_component_full_state_replayed' if passed else 'prefill128_commit_component_failure_preserved',
                'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'full_state_replay':replayed,
                'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'files_verified':result['files_verified']}));return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
