"""Replay complete T128 attention outputs, actual GPU controls and owner identity."""
from pathlib import Path
import argparse,hashlib,json,statistics,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.bf16_prefill128_attention_oracle import case_plan,compare_full,ATOL,RTOL
from test2026.cuda.qwen38_5090.run_bf16_prefill128_attention_component import JOB,REFERENCE
sha=lambda raw:hashlib.sha256(raw).hexdigest()
GPU='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
WORDS=128*24*256

def inventory(records):
    plan=case_plan()
    assert len(records)==74 and records[0]['event']=='gpu_identity'
    assert records[0]['sm']==120 and records[0]['uuid']==GPU
    assert records[-1]=={'status':'prefill128_attention_collected_needs_full_fp64_replay','cases':36,'witness_bytes':36*WORDS*4,'performance_admitted':False,'model_qualified':False}
    result=[]
    for i,case in enumerate(plan):
        started,row=records[1+2*i:3+2*i]
        assert started=={'event':'case_started',**case}
        assert row['event']=='case_collected' and all(type(row[k]) is int and row[k]==v for k,v in case.items())
        for k in ('witness_offset','witness_bytes','rejected_groups','rejected_invocations','full_output_words'):
            assert type(row[k]) is int
        assert row['witness_offset']==i*WORDS*4 and row['witness_bytes']==WORDS*4 and row['full_output_words']==WORDS
        assert row['samples']==[]
        assert row['rejected_groups']==(65 if case['id'] in (0,14) else 0)
        assert row['rejected_invocations']==(128 if case['id'] in (0,14) else 0)
        assert row['readonly_exact'] is row['canaries_exact'] is True
        assert row['prefix_cache_exact'] is bool(case['end'])
        assert row['graph_replay_exact'] is (case['id'] in (1,15,127))
        result.append((case,row))
    return result

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--proofs',type=Path,required=True);args=a.parse_args()
    archive=args.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/';prefix='Garnet/test2026/cuda/qwen38_5090/'
    numerical=[];failures=[];memory=None
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name))
        manifest=doc('evidence-manifest.json')
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,digest in manifest['files'].items():assert sha(z.read(name))==digest
        w=doc(base+'workflow.json');praw=z.read(prefix+'evidence/prefill128-attention-component-preparation.json');p=json.loads(praw)
        assert sha(praw)==w['preparation_sha256'] and w['cleanup_finished'] is True and type(w['exit_code']) is int
        assert w['runtime_loaded'] is w['performance_admitted'] is False
        assert w['source_sha256']==w['source_after_sha256']==p['files']
        assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
        assert w['environment']==w['environment_after']==p['environment'] and w['sdk_before']==w['sdk_after']
        for name,digest in p['files'].items():assert sha(z.read(name))==digest
        for name in ('bf16_prefill128_attention_oracle.py','bf16_kv_overlay_oracle.py','verify_bf16_prefill128_attention_component.py','run_bf16_prefill128_attention_component.py'):
            assert sha(Path(__file__).with_name(name).read_bytes())==p['files'][prefix+name]
        profiled=p['profiled_parent'];raw=z.read('results/'+profiled['job']+'-verified.json');proof=json.loads(raw)
        assert sha(raw)==profiled['proof_sha256'] and proof['status']=='native_cuda_stdio_profiled_replayed'
        assert sha((args.proofs/(profiled['job']+'-evidence.zip')).read_bytes())==proof['archive_sha256']==profiled['archive_sha256']
        parent=p['reference_parent'];raw=z.read('results/'+REFERENCE+'-verified.json');proof=json.loads(raw)
        assert sha(raw)==parent['proof_sha256'] and proof['status']=='reference_fixed4k_vllm_observed_profile_replayed'
        assert proof['archive_sha256']==w['closed_reference']['archive_sha256']==parent['archive_sha256']==sha((args.proofs/(REFERENCE+'-evidence.zip')).read_bytes())
        with zipfile.ZipFile(args.proofs/(REFERENCE+'-evidence.zip')) as previous:
            closed=json.loads(previous.read('results/'+REFERENCE+'/workflow.json'))
            assert closed==doc('results/'+REFERENCE+'/workflow.json')
            assert type(closed['exit_code']) is int and closed['exit_code']==0 and closed['cleanup_finished'] is True
            assert closed['server_exit_code']==0 and closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
        passed=w['exit_code']==0
        if passed:
            assert w['status']=='prefill128_attention_component_collected_needs_replay'
            for name in ('cpu-ledger','build-ledger','gpu-ledger'):
                events=[json.loads(line) for line in z.read(base+name+'/commands.jsonl').splitlines()]
                assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
                assert type(events[1]['exit_code']) is int and events[1]['exit_code']==0
            assert b'Ran 11 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
            assert sha(z.read('tools/bf16-kv-native/prefill128-attention-probe.exe'))==w['executable_sha256']
            records=[]
            for line in z.read(base+'gpu-ledger/commands.log').splitlines():
                if line.startswith(b'{'):
                    row=json.loads(line)
                    if row.get('event') in ('gpu_identity','case_started','case_collected') or row.get('status')=='prefill128_attention_collected_needs_full_fp64_replay':records.append(row)
            pairs=inventory(records);binary=z.read(base+'native/witnesses.f32');assert len(binary)==36*WORDS*4
            for case,row in pairs:
                offset=row['witness_offset']
                try:numerical.append({'id':case['id'],**compare_full(case,binary[offset:offset+WORDS*4])})
                except ValueError as error:failures.append({'id':case['id'],'error':str(error)})
            resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()]
            assert type(resource['child_exit_code']) is int and resource['child_exit_code']==0 and resource['gpu']['uuid']==GPU
            gaps=[b['monotonic']-a['monotonic'] for a,b in zip(samples,samples[1:])];assert samples and all(v>0 for v in gaps)
            memory=resource['sampled_memory'];assert memory['samples']==len(samples) and memory['sampled_peak_adapter_used_bytes']==max(row['used_bytes'] for row in samples)
            assert memory['observed_sample_gap_max_seconds']==(max(gaps) if gaps else None)
            assert memory['observed_sample_gap_median_seconds']==(statistics.median(gaps) if gaps else None)
        passed=passed and not failures
        result={'status':'prefill128_attention_component_full_fp64_replayed' if passed else 'prefill128_attention_component_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'cases':numerical,'numerical_failures':failures,'atol':ATOL,'rtol':RTOL,'full_output_scalars_replayed':sum(row['full_output_scalars'] for row in numerical),'correctness_sampled_memory':memory,'graph_launches':9 if passed else None,'malformed_invocations':256 if passed else None,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(prefix+Path(__file__).name)),'runtime_integrated':False,'model_or_serving_qualified':False,'performance_admitted':False,'two_times_target_achieved':False,'scope':p['scope']}
    target=args.proofs/(JOB+'-verified.json');raw=json.dumps(result,indent=2).encode()
    assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'full_output_scalars_replayed':result['full_output_scalars_replayed'],'numerical_failures':failures}))
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
