"""Prepare a complete matched vLLM trial with a separate observed-output policy."""
from pathlib import Path
import argparse,ast,hashlib,io,json,os,re,sys,unittest,zipfile
sha=lambda b:hashlib.sha256(b).hexdigest()
METHODS=('run_vllm_observed_fixed4k.py','verify_vllm_observed_fixed4k.py','prepare_vllm_observed_fixed4k_run.py',
         'vllm_observed_component_parent.py','test_vllm_observed_fixed4k_owner.py','reference_vllm_observed_output.py',
         'benchmark_vllm_observed_output.py','probe_vllm_observed_fixed4k.py','test_reference_vllm_observed_output.py',
         'test_vllm_observed_output_benchmark_http.py','evidence/vllm-observed-output-client-derivation.json',
         'evidence/vllm-observed-fixed4k-pipeline-derivation.json')
REQUIRED=('serve_vllm_fixed4k.sh','reference_vllm_observed_output.py','benchmark_vllm_observed_output.py','probe_vllm_observed_fixed4k.py',
          'run_vllm_observed_fixed4k.py','verify_vllm_observed_fixed4k.py','vllm_observed_component_parent.py','prepare_vllm_fixed4k_metadata.py',
          'inspect_vllm_fixed4k_config_linux.py','inspect_reference_fixed4k_linux.py','vllm_cache_control.py','run_reference_greedy_candidate.py',
          'stop_reference_server.py','reference_fixed4k_vllm_boundary.py','evidence/fixed-length-4096-corpus.zip')

def require_observed_dependencies(files):
    for n in REQUIRED:
        value=files.get('Garnet/test2026/cuda/qwen38_5090/'+n)
        if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
            raise ValueError('Observed vLLM runtime dependency missing or invalid: '+n)

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    from vllm_observed_component_parent import admit_component,JOB
    observation='vllm-output-characterization-attempt1';native='fixed-length-http-study-attempt1'
    parents={};workflow=None
    for name,status in ((observation,'vllm_output_characterization_replayed'),(JOB,'prefill128_commit_component_full_state_replayed'),(native,'fixed_length_http_study_replayed')):
        raw=(args.proofs/(name+'-verified.json')).read_bytes();q=json.loads(raw)
        assert raw==(args.proofs/(name+'-remote-verified.json')).read_bytes() and q['status']==status
        archive=(args.proofs/(name+'-evidence.zip')).read_bytes();assert sha(archive)==q['archive_sha256']
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
            for n,h in manifest['files'].items():assert sha(z.read(n))==h
            w=json.loads(z.read('results/'+name+'/workflow.json'));assert type(w['exit_code']) is int and w['exit_code']==0 and w['cleanup_finished'] is True
            assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
            if name==observation:workflow=w;assert type(w['server_exit_code']) is int
        parents[name]={'job':name,'status':status,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']}
        if name==JOB:component=admit_component(archive,raw)
    assert workflow is not None;files=dict(workflow['source_sha256'])
    for n in METHODS:
        key='Garnet/test2026/cuda/qwen38_5090/'+n;digest=sha((m/n).read_bytes());assert key not in files or files[key]==digest;files[key]=digest
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    for n,row in parents.items():
        key='results/'+n+'-verified.json';assert key not in files or files[key]==row['proof_sha256'];files[key]=row['proof_sha256']
    require_observed_dependencies(files)
    candidate=json.loads((m/'evidence/bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for n,h in files.items():
        if n.startswith('Garnet/'):
            relative=n.removeprefix('Garnet/');assert sha((repo/relative).read_bytes())==candidate.get(relative,h),('Qualified baseline source drift',n)
    inherited=json.loads((m/'evidence/vllm-output-characterization-run-preparation.json').read_bytes())
    modules=[*inherited['test_modules'],*('test2026.cuda.qwen38_5090.'+n for n in ('test_reference_vllm_observed_output','test_vllm_observed_output_benchmark_http','test_vllm_observed_fixed4k_owner'))]
    assert all('Garnet/'+n.replace('.','/')+'.py' in files for n in modules)
    loader=unittest.TestLoader();suite=loader.loadTestsFromNames(modules);assert not loader.errors;count=suite.countTestCases()
    import test2026.cuda.qwen38_5090.run_logged as ledger
    ledger.source_identity=lambda:{'scope':'explicit owned observed-baseline inputs; unrelated untracked files excluded','owned_source_sha256':files}
    os.chdir(repo);code=ledger.run([sys.executable,'-m','unittest',*modules],args.work/'vllm-observed-fixed4k-cpu-ledger','correctness',
        'All reference identity/length/cache/PID and observed variant/HTTP/source/closed-component/derivation/dependency controls; CPU fixtures only, no GPU/server',
        [m/n for n in METHODS]);assert code==0
    scope='One exclusive ordinary vLLM0.32.0 WSL matched actual3968+128 reference after both-machine288request isolation diagnosis and latest T128 component closure. Same checkpoint/modelopt_mixed/BF16KV/F32SSM/4096total/zeroreservedslots/4GiBcache/max8sequences/decodegraphs1/2/4/8/TP1/cache-off profile. Actual ready JSON/version/installed74sources/eight checkpoint/fiveconfig/PID verification. Two full4K boundaries, fourVLM/16short IDs, fresh805adapted screen,124serial actual prompt/completion-ID controls,124warmup/timed rawstreams in eight C1/2/4/8text/image cohorts. Preserve all text variants with record_vllm_stream_text_variants_v1 sidecars after timing/sampling; never infer streamed IDs, numerical harmlessness or quality equivalence. Original strict failure untouched. Full raw metrics/cache/memory/source/runtime/owned cleanup replay required before any chart promotion. Strongest tuned/MTP/repeats/INT4/4B/native optimization/overall2x remain open.'
    p={'status':'vllm_observed_fixed4k_prepared_unrun','files':files,'runtime_dependencies':workflow['runtime_dependencies'],
       'native_proof_sha256':parents[native]['proof_sha256'],'native_archive_sha256':parents[native]['archive_sha256'],
       'observation_proof_sha256':parents[observation]['proof_sha256'],'observation_archive_sha256':parents[observation]['archive_sha256'],
       'component_parent':component,'expected_cpu_tests':count,'test_modules':modules,'scope':scope}
    target=m/'evidence/vllm-observed-fixed4k-run-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha((m/n).read_bytes())} for n in METHODS]
    rows.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(raw)})
    plan=args.work/'vllm-observed-fixed4k-upload-plan.json';data=json.dumps({'uploads':rows,'scope':scope},indent=2).encode();assert not plan.exists() or plan.read_bytes()==data
    if not plan.exists():plan.write_bytes(data)
    print(json.dumps({'status':p['status'],'source_pins':len(files),'cpu_tests':count,'targets':len(rows),'preparation_sha256':sha(raw),'gpu_executed':False}))

if __name__=='__main__':raise SystemExit(main())
