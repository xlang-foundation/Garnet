"""Pin standalone T128 commit qualification after the actual CUDA phase diagnosis."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,zipfile
sha=lambda b:hashlib.sha256(b).hexdigest()
METHODS=('bf16_prefill128_commit_oracle.py','test_bf16_prefill128_commit_oracle.py','test_bf16_prefill128_component_owner.py',
         'probe_bf16_prefill128_commit.cu','prefill128_commit_fixture.h','qwen38_bf16_prefill128_commit.cu',
         'qwen38_bf16_prefill128_commit.cuh','bf16_prefill128_layout.py','bf16_low_batch_layout.py','test_bf16_prefill128_layout.py',
         'run_bf16_prefill128_commit_component.py','verify_bf16_prefill128_commit_component.py','prepare_bf16_prefill128_commit_component.py',
         'build_bf16_prefill128_commit_component.bat','monitor_native_probe.py','sample_vram.py','run_logged.py','run_bf16_low_batch_native_build.py',
         'evidence/prefill128-commit-probe-layout.json','evidence/nvfp4-config.json','evidence/bf16-scores-tiled-native-api-build-preparation.json')

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];job='native-cuda-stdio-profiled-bootstrap-attempt1'
    raw=(args.proofs/(job+'-verified.json')).read_bytes();q=json.loads(raw)
    assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes() and q['status']=='native_cuda_stdio_profiled_replayed'
    archive=args.proofs/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        w=json.loads(z.read('results/'+job+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] is True and w['library_unchanged'] is True
        assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    files={'Garnet/test2026/cuda/qwen38_5090/'+n:sha((m/n).read_bytes()) for n in METHODS}
    for n in ('src/cuda/qwen38_bf16_kv_commit.cuh','xModel/qwen35/vl_0_8b/bf16_low_batch_kv_profile.py'):
        files['Garnet/'+n]=sha((repo/n).read_bytes())
    files['results/'+job+'-verified.json']=sha(raw)
    for n in METHODS:
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    modules=['test2026.cuda.qwen38_5090.'+n for n in ('test_bf16_prefill128_layout','test_bf16_prefill128_commit_oracle','test_bf16_prefill128_component_owner')]
    sys.path[:0]=[str(repo),str(m)];import test2026.cuda.qwen38_5090.run_logged as ledger
    # The separate private untracked probe is outside this task. Do not read it
    # through the generic ledger's default untracked-source inventory either.
    ledger.source_identity=lambda:{'scope':'explicit owned component inputs; unrelated untracked files excluded','owned_source_sha256':files}
    os.chdir(repo);code=ledger.run([sys.executable,'-m','unittest',*modules],args.work/'prefill128-commit-component-cpu-ledger','correctness',
        'Actual10CPU layout/codec/BF16/untouched-slot/closed-reference/full-inventory controls; no local GPU or target executable',[m/n for n in METHODS]);assert code==0
    environment=json.loads((m/'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['environment']
    p={'status':'prefill128_commit_component_prepared_gpu_unrun','files':files,'runtime_dependencies':w['runtime_dependencies'],'environment':environment,
       'profiled_parent':{'job':job,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']},'test_modules':modules,'cpu_tests':10,
       'scope':'Exclusive standalone B1/T128 final-state commit component after actual vLLM diagnosis supervisor terminal/closed archive and idle guards. Rebuild/link using pinned remote tools; record actual compiler version rather than infer it from folder names or earlier summaries.16valid/35invalid complete canonical states include guards and independent CPU integer BF16 rounding, full target input/final snapshots and per-call readonly checks,3replays of one captured two-node graph. Lossless full-state RLE and independent NumPy replay required. SDK/runtime/Garnet DLL/default source preserved. No full T128 forward/runtime/model/HTTP/speed or2x qualification.'}
    target=m/'evidence/prefill128-commit-component-preparation.json';data=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==data
    if not target.exists():target.write_bytes(data)
    rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha((m/n).read_bytes())} for n in METHODS]
    rows.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(data)})
    plan=args.work/'prefill128-commit-component-upload-plan.json';blob=json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode();assert not plan.exists() or plan.read_bytes()==blob
    if not plan.exists():plan.write_bytes(blob)
    print(json.dumps({'status':p['status'],'source_pins':len(files),'cpu_tests':10,'preparation_sha256':sha(data),'gpu_executed':False}))

if __name__=='__main__':raise SystemExit(main())
