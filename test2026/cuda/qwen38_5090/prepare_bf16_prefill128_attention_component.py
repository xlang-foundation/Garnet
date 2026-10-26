"""Prepare isolated T128 attention math qualification after the closed 4K baseline."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS=('bf16_prefill128_attention_profile.py','bf16_prefill128_attention_oracle.py',
 'test_bf16_prefill128_attention_profile.py','test_bf16_prefill128_attention_oracle.py','test_bf16_prefill128_attention_owner.py',
 'qwen38_bf16_prefill128_attention.cu','qwen38_bf16_prefill128_attention.cuh','probe_bf16_prefill128_attention.cu',
 'prefill128_commit_fixture.h','bf16_prefill128_layout.py','bf16_low_batch_layout.py','bf16_kv_overlay_oracle.py','probe_bf16_scores_tiled.cu',
 'run_bf16_prefill128_attention_component.py','build_bf16_prefill128_attention_component.bat',
 'verify_bf16_prefill128_attention_component.py','prepare_bf16_prefill128_attention_component.py',
 'monitor_native_probe.py','sample_vram.py','run_logged.py','run_bf16_low_batch_native_build.py',
 'evidence/prefill128-attention-component-derivation.json','evidence/prefill128-attention-probe-derivation.json',
 'evidence/prefill128-attention-owner-derivation.json','evidence/prefill128-attention-source-overlay.cu.txt','evidence/nvfp4-config.json','evidence/bf16-scores-tiled-native-api-build-preparation.json')

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2]
    parents={};workflow=None
    for job,status in (('native-cuda-stdio-profiled-bootstrap-attempt1','native_cuda_stdio_profiled_replayed'),('vllm-observed-fixed4k-ordinary-attempt1','reference_fixed4k_vllm_observed_profile_replayed')):
        raw=(args.proofs/(job+'-verified.json')).read_bytes();q=json.loads(raw)
        assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes() and q['status']==status
        archive=args.proofs/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            w=json.loads(z.read('results/'+job+'/workflow.json'))
            assert type(w['exit_code']) is int and w['exit_code']==0 and w['cleanup_finished'] is True
            assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
        parents[job]={'job':job,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']}
        if job.startswith('vllm-'):workflow=w
    files={'Garnet/test2026/cuda/qwen38_5090/'+n:sha((m/n).read_bytes()) for n in METHODS}
    extra=('src/cuda/qwen35_attention_core.cuh','src/cuda/qwen38_bf16_kv_commit.cuh','xModel/qwen35/vl_0_8b/bf16_low_batch_kv_profile.py')
    for name in extra:files['Garnet/'+name]=sha((repo/name).read_bytes())
    for job,parent in parents.items():files['results/'+job+'-verified.json']=parent['proof_sha256']
    for name in METHODS:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    d=json.loads((m/'evidence/prefill128-attention-owner-derivation.json').read_bytes())
    previous=(m/d['source']).read_bytes();assert sha(previous)==d['source_sha256'];derived=previous.decode()
    for step in d['replacements']:
        assert derived.count(step['old'])==step['count'];derived=derived.replace(step['old'],step['new'])
    assert derived.encode()==(m/d['target']).read_bytes() and sha(derived.encode())==d['target_sha256']
    modules=['test2026.cuda.qwen38_5090.'+name for name in ('test_bf16_prefill128_attention_profile','test_bf16_prefill128_attention_oracle','test_bf16_prefill128_attention_owner')]
    sys.path[:0]=[str(repo),str(m)]
    import test2026.cuda.qwen38_5090.run_logged as ledger
    ledger.source_identity=lambda:{'scope':'explicit owned attention component inputs; unrelated untracked files excluded','owned_source_sha256':files}
    os.chdir(repo)
    code=ledger.run([sys.executable,'-m','unittest',*modules],args.work/'prefill128-attention-component-cpu-ledger','correctness',
        'Actual11CPU geometry, full FP64 oracle, analytic/sampled controls, complete recorded cases and closed successful reference; no local GPU or target executable',[m/n for n in METHODS])
    assert code==0
    environment=json.loads((m/'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['environment']
    p={'status':'prefill128_attention_component_prepared_gpu_unrun','files':files,'runtime_dependencies':workflow['runtime_dependencies'],'environment':environment,
       'profiled_parent':parents['native-cuda-stdio-profiled-bootstrap-attempt1'],'reference_parent':parents['vllm-observed-fixed4k-ordinary-attempt1'],
       'test_modules':modules,'cpu_tests':11,'scope':'Exclusive standalone B1/T128 BF16 canonical attention component. Same scores/softmax/context math as the retained B1 implementation; separate exact T128 profile, explicit plain/tiled strategy and alignment guards.36 declared cases cover all16full-attention layers, empty/partial/full128 and page/cache limits. Raw all786432F32 outputs per case independently replay against full FP64 attention (ATOL3e-5/RTOL2e-5); inactive exact zero. Source-pinned C++ also checks full plain/tiled probability/output bits, three graphs with3launches each, cached-prefix last-token identity, complete canonical/inputs readonly and canaries; probability/canonical buffers are not serialized.256actual malformed invocations. Fresh remote SM120 build/link, compiler version/SDK/source/runtime and exclusive cleanup retained. No production DLL replacement, full T128 forward/runtime/model/HTTP/performance or2x admission.'}
    target=m/'evidence/prefill128-attention-component-preparation.json';raw=json.dumps(p,indent=2).encode()
    assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha((m/n).read_bytes())} for n in METHODS]
    rows.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(raw)})
    plan=args.work/'prefill128-attention-component-upload-plan.json';blob=json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode()
    assert not plan.exists() or plan.read_bytes()==blob
    if not plan.exists():plan.write_bytes(blob)
    print(json.dumps({'status':p['status'],'source_pins':len(files),'cpu_tests':11,'preparation_sha256':sha(raw),'gpu_executed':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
