"""Freeze a separate vLLM output/isolation diagnosis after the actual strict failure."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile

METHODS=('vllm_characterization_parent.py','run_vllm_output_characterization.py',
         'verify_vllm_output_characterization.py','prepare_vllm_output_characterization_run.py',
         'test_vllm_characterization_pipeline.py','vllm_concurrency_characterization.py',
         'replay_vllm_concurrency_characterization.py','test_vllm_concurrency_characterization.py',
         'verify_vllm_fixed4k_attempt3_partial.py','evidence/vllm-output-characterization-client-derivation.json',
         'evidence/vllm-output-characterization-pipeline-derivation.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def require_linked_proofs(files,parents):
    for name,row in parents.items():
        key='results/'+name+'-verified.json'
        if files.get(key)!=row['proof_sha256']:
            raise ValueError('Inherited proof must be pinned and archived: '+name)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    from vllm_characterization_parent import admit,PARENT
    from prepare_vllm_fixed4k_run import require_runtime_dependencies
    from run_logged import run
    read=lambda n:(args.proofs/n.removeprefix('results/')).read_bytes()
    parent=admit(read)
    for suffix in ('','-partial','-quality'):
        assert read('results/'+PARENT+suffix+'-verified.json')==(args.proofs/(PARENT+suffix+'-remote-verified.json')).read_bytes()
    with zipfile.ZipFile(args.proofs/(PARENT+'-evidence.zip')) as z:
        w=json.loads(z.read('results/'+PARENT+'/workflow.json'))
        inherited=json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/vllm-fixed4k-launcher-completed-preparation.json'))
        assert inherited['files']==w['source_sha256'] and inherited['runtime_dependencies']==w['runtime_dependencies']
    files=dict(inherited['files'])
    for n in METHODS:
        key='Garnet/test2026/cuda/qwen38_5090/'+n;digest=sha((m/n).read_bytes())
        assert key not in files or files[key]==digest
        files[key]=digest
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    for suffix in ('','-partial','-quality'):
        key='results/'+PARENT+suffix+'-verified.json';files[key]=sha(read(key))
    for parent_name,row in inherited['parents'].items():
        key='results/'+parent_name+'-verified.json';raw=read(key);q=json.loads(raw)
        assert raw==(args.proofs/(parent_name+'-remote-verified.json')).read_bytes()
        assert sha(raw)==row['proof_sha256'] and q['status']==row['status']
        assert sha(read('results/'+parent_name+'-evidence.zip'))==q['archive_sha256']==row['archive_sha256']
        assert key not in files or files[key]==sha(raw)
        files[key]=sha(raw)
    require_linked_proofs(files,inherited['parents'])
    candidate=json.loads((m/'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for n,h in files.items():
        if n.startswith('Garnet/'):
            relative=n.removeprefix('Garnet/');assert sha((repo/relative).read_bytes())==candidate.get(relative,h),('Pinned source drift',n)
    require_runtime_dependencies(files)
    modules=[*inherited['test_modules'],'test2026.cuda.qwen38_5090.test_vllm_concurrency_characterization','test2026.cuda.qwen38_5090.test_vllm_characterization_pipeline']
    assert all('Garnet/'+n.replace('.','/')+'.py' in files for n in modules)
    loader=unittest.TestLoader();suite=loader.loadTestsFromNames(modules);assert not loader.errors;count=suite.countTestCases();os.chdir(repo)
    code=run([sys.executable,'-m','unittest',*modules],args.work/'vllm-characterization-pipeline-cpu-ledger','correctness',
             'Synthetic288HTTP streams/IDs/overlap/color and strict failed-parent/runtime/source/config/owner/derivation controls; no model or GPU',
             [m/n for n in METHODS]);assert code==0
    scope='One exclusive ordinary vLLM0.32.0 WSL output/isolation diagnosis after actual strict4K trial3 failure and both-machine partial/805screen replays. Same pinned checkpoint/4096total context, BF16KV/F32SSM, max8sequences, full decodegraphs1/2/4/8, prefix/MM caches off and owned PID/config/installed sources.192actual nonstream IDs plus64full3968+128 and32color streams in135overlap waves. Retain all concurrent/rotated/serial-before/after text/ID variants and color/spatial isolation. Never infer streamed IDs, numerical cause/harmlessness, quality equivalence or a benchmark. Original strict failure remains. Future observed-output policy needs actual closed/replayed diagnosis and separate full trial. Native/tuned/MTP/repeated/INT4/4B/2x objectives remain open.'
    p={'status':'vllm_output_characterization_prepared_unrun','preparation_version':'output_diagnosis_v1',
       'files':files,'runtime_dependencies':inherited['runtime_dependencies'],'parents':inherited['parents'],
       'native_proof_sha256':inherited['native_proof_sha256'],'native_archive_sha256':inherited['native_archive_sha256'],
       'diagnosis_parent':parent,'test_modules':modules,'expected_cpu_tests':count,'scope':scope}
    target=m/'evidence/vllm-output-characterization-run-preparation.json';raw=json.dumps(p,indent=2).encode()
    assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    uploads=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha((m/n).read_bytes())} for n in METHODS]
    uploads.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(raw)})
    plan=args.work/'vllm-characterization-pipeline-upload-plan.json';data=json.dumps({'uploads':uploads,'scope':scope},indent=2).encode()
    assert not plan.exists() or plan.read_bytes()==data
    if not plan.exists():plan.write_bytes(data)
    print(json.dumps({'status':p['status'],'source_pins':len(files),'targets':len(uploads),'cpu_controls':count,'preparation_sha256':sha(raw),'gpu_or_server_executed':False}))
    return 0

if __name__=='__main__':raise SystemExit(main())
