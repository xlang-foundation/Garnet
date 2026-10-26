"""Preserve rejected static-runtime assumption; no old result is relabeled as qualified."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_reconstruct import sdk_case
JOB='z128-reconstruct-runtime-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        p=doc(M+'evidence/z128-reconstruct-runtime-preparation.json');w=doc(base+'workflow.json');pins={**p['files'],**p['methods']};assert w['exit_code']==1 and w['error']=='AssertionError()' and w['cleanup_finished'] and w['source_before']==w['source_after']==pins and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and not w['production_source_or_library_modified']
        report=doc(base+'trace-on/report.json');r=report['native_report'];stderr=z.read(base+'trace-on/stderr.log');assert report['exit_code']==0 and report['replay']==sdk_case(r,stderr,True) and base+'trace-unset-ledger/commands.jsonl' not in manifest['files'] and base+'trace-on/runtime-origin.json' not in manifest['files']
        assert r['cuda_runtime_linkage']=='static' and (r['cuda_runtime_header_version'],r['cuda_runtime_version'],r['cuda_driver_version'])==(13020,13040,13040) and r['cuda_runtime_status']==r['cuda_driver_status']==0
        mapping=z.read(base+'sdk-build/constructor.map').decode('ascii',errors='replace');origins=[line.strip() for line in mapping.splitlines() if 'cudaGetDevice' in line and 'cudart:generated_cuda_runtime_api_loader.obj' in line];assert origins
        dependencies=z.read(base+'sdk-build/dependencies.txt').decode('ascii',errors='replace');assert 'cublasLt64_13.dll' in dependencies and 'cudart64_13.dll' not in dependencies
        result={'status':'SDK_origin_observation_header_runtime_equality_assumption_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'source_runtime_SDK_unchanged':True,'native_constructor_exit_code':0,'owner_exit_code':1,'original_static_label_preserved':r['cuda_runtime_linkage'],'header_version':13020,'runtime_version':13040,'driver_version':13040,'linker_API_loader_origins':origins,'SDK_configuration_and_M32_M128_checks_observed':True,'trace_unset_process_executed':False,'old_runtime_identity_qualified':False,'documentation':'https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html','scope':'NVIDIA documents Windows CUDA13 runtime from display driver; header need not match runtime. Original exact equality/static-implementation assumption rejected and retained. A new hybrid-specific pinned version/map/source/runtime observation remains required. No old field is rewritten or promoted.','actual_Garnet_plugin_rewired':False,'GPU_model_or_matmul_invoked':False,'numerical_or_model_or_quality_or_performance_qualified':False,'analyzer_sha256':sha(Path(__file__).read_bytes())}
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-failure-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw)}))
if __name__=='__main__':main()
