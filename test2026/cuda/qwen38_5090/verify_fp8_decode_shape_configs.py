"""Independent raw SDK query/failure replay; not numerical/performance admission."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fp8_decode_shape_configs import replay
JOB='fp8-decode-shape-configs-attempt1';P='Garnet/test2026/cuda/qwen38_5090/';BASE='results/'+JOB+'/';sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);args=a.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        mf=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(mf['files'])+1
        for n,h in mf['files'].items():assert sha(z.read(n))==h
        raw=z.read(P+'evidence/fp8-decode-shape-configs-preparation.json');p=json.loads(raw);w=json.loads(z.read(BASE+'workflow.json'));assert sha(raw)==w['preparation_sha256'] and w['cleanup_finished'] and w['unchanged_before_after']
        for a,b,key in [('source_before','source_after','files'),('runtime_before','runtime_after','runtime'),('environment_before','environment_after','environment'),('checkpoint_before','checkpoint_after','checkpoints'),('cuda_objects_before','cuda_objects_after','cuda_objects')]:assert w[a]==w[b]==p[key]
        assert w['cache_before_sha256']==w['cache_after_sha256']==p['cache_before_sha256'] and len(p['cuda_objects'])==34 and w['production_source_or_DLL_installed'] is w['matmul_model_or_benchmark_executed'] is False
        for n,h in p['methods'].items():assert sha(z.read(n))==h
        for n in ('verify_fp8_decode_shape_configs.py','fp8_decode_shape_configs.py'):assert Path(__file__).with_name(n).read_bytes()==z.read(P+n)
        for job,row in p['parents'].items():
            proof=z.read('results/'+job+'-verified.json');q=json.loads(proof);assert proof==(args.proofs/(job+'-verified.json')).read_bytes() and sha(proof)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256']==sha((args.proofs/(job+'-evidence.zip')).read_bytes())
        stages={}
        for name in ('cpu','build','SDK'):
            file=BASE+name+'-ledger/commands.jsonl'
            if file not in mf['files']:continue
            rows=[json.loads(s) for s in z.read(file).splitlines()];assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished' and rows[0]['source']['owned_source_sha256']==rows[1]['source_after']['owned_source_sha256']==p['files'];stages[name]=rows[1]['exit_code'];assert stages[name]==w[name+'_exit_code']
        analysis=None
        if w['exit_code']==0:
            assert stages==dict(cpu=0,build=0,SDK=0) and w['status']=='actual_FP8_decode_shape_configuration_query_collected_no_matmul';assert b'Ran 3 tests' in z.read(BASE+'cpu-ledger/commands.log')
            report=json.loads(z.read(BASE+'SDK-stdout.json'));assert report==json.loads(z.read(BASE+'report.json'));analysis=replay(report,p['gpu_uuid']);assert analysis==json.loads(z.read(BASE+'analysis.json')) and analysis['all_known_M4_references_match']
            imports=z.read(BASE+'private-build/imports.txt');names={line.strip().split()[-1] for line in imports.splitlines() if line.strip()};assert b'cublasLtMatmul' not in names and b'cudaMalloc' not in names and b'garnet.dll' not in imports.lower() and b'nvinfer' not in imports.lower() and b'cublasLtMatmulAlgoGetHeuristic' in imports and b'cublasLtMatmulAlgoCheck' in imports
        else:assert w['status']=='failed' and isinstance(w['error'],str)
        result=dict(status='actual_FP8_decode_shape_queries_replayed_no_matmul' if w['exit_code']==0 else 'FP8_decode_shape_query_failure_replayed',exit_code=w['exit_code'],error=w.get('error'),archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,archive_members=len(z.namelist()),stages=stages,analysis=analysis,source_runtime_SDK_config_index_all34objects_CMake_unchanged=True,model_matmul_or_performance_qualified=False,original_failed_model_job_promoted=False,replay_verifier_sha256=sha(Path(__file__).read_bytes()),scope=p['scope'])
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw);print(json.dumps(dict(status=result['status'],exit_code=result['exit_code'],proof_sha256=sha(raw),analysis=analysis)));return 0
if __name__=='__main__':raise SystemExit(main())
