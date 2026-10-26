from pathlib import Path
import argparse,hashlib,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.z128_reconstruct import sdk_case,replay
from test2026.cuda.qwen38_5090.z128_reconstruct_runtime import identity
JOB='z128-reconstruct-runtime-attempt1';M='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');base='results/'+JOB+'/'
    with zipfile.ZipFile(archive) as z:
        doc=lambda name:json.loads(z.read(name));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        raw=z.read(M+'evidence/z128-reconstruct-runtime-preparation.json');p=json.loads(raw);w=doc(base+'workflow.json');pins={**p['files'],**p['methods']};assert sha(raw)==w['preparation_sha256'] and w['source_before']==w['source_after']==pins and w['runtime_before']==w['runtime_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['cleanup_finished'] and not w['production_source_or_library_modified'] and not w['GPU_model_or_matmul_invoked']
        for name,h in p['methods'].items():assert sha(z.read(name))==h
        assert sha(Path(__file__).read_bytes())==p['methods'][M+Path(__file__).name]
        parent='z128-reconstruct-construction-attempt2';proof=z.read('results/'+parent+'-quarantine-verified.json');q=json.loads(proof);assert sha(proof)==p['parent_proof_sha256'] and proof==(a.proofs/(parent+'-quarantine-verified.json')).read_bytes() and sha((a.proofs/(parent+'-evidence.zip')).read_bytes())==p['parent_archive_sha256'] and q['CPU_processes']==1400 and not q['active_CUDA_runtime_identity_established']
        phases={}
        for phase in ('cpu-source','CPU-parent-replay','sdk-build','trace-on','trace-unset'):
            key=base+phase+'-ledger/commands.jsonl'
            if key not in manifest['files']:continue
            rows=[json.loads(line) for line in z.read(key).splitlines()];assert len(rows)==2 and rows[0]['event']=='started' and rows[1]['event']=='finished' and rows[0]['source']['owned_source_sha256']==rows[1]['source_after']['owned_source_sha256']==pins;phases[phase]={'exit_code':rows[1]['exit_code']}
            if phase in ('trace-on','trace-unset'):
                report=doc(base+phase+'/report.json');stdout=z.read(base+phase+'/stdout.json');stderr=z.read(base+phase+'/stderr.log');assert report['exit_code']==rows[1]['exit_code'] and report['stdout_sha256']==sha(stdout) and report['stderr_sha256']==sha(stderr)
                phases[phase]['trace']=replay(stderr) if phase=='trace-on' else None
                if rows[1]['exit_code']==0:
                    assert report['native_report']==json.loads(stdout) and report['replay']==sdk_case(report['native_report'],stderr,phase=='trace-on');r=identity(report['native_report'],'D:/CantorAI/5090-study-20261023/results/'+JOB+'/sdk-build/constructor.exe',z.read(base+'sdk-build/dependencies.txt'),z.read(base+'sdk-build/constructor.map'));origin=doc(base+phase+'/runtime-origin.json');assert origin['identity']==r;phases[phase]['runtime_identity']=origin
                    for name,row in origin['observed_module_files'].items():
                        if name=='cublasLt64_13.dll':assert row['sha256']==p['runtime'][name]
                        elif name=='cudaGetDevice':assert row['sha256']==sha(z.read(base+'sdk-build/constructor.exe'))
                        else:assert name=='nvcuda.dll' and row['sha256'] and row['path'].replace(chr(92),'/').lower().startswith('c:/windows/')
                    runtime_keys=[name for name in p['environment'] if name.replace(chr(92),'/').lower().endswith('/lib/x64/cudart.lib')];assert len(runtime_keys)==1 and origin['static_CUDART_library_sha256']==p['environment'][runtime_keys[0]]
                resource=doc(base+phase+'-resources/resource-monitor.json');assert resource['gpu']['uuid']==p['gpu_uuid'] and resource['child_exit_code']==rows[1]['exit_code']
        passed=w['exit_code']==0
        if passed:
            assert len(phases)==5 and all(row['exit_code']==0 for row in phases.values()) and w['status']=='separate_reconstruction_SDK_runtime_origin_collected' and b'Ran 2 tests' in z.read(base+'cpu-source-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-source-ledger/commands.log') and sha(z.read(base+'sdk-build/constructor.exe'))==w['SDK_executable_sha256']
            messages=[json.loads(line) for line in z.read(base+'CPU-parent-replay-ledger/commands.log').splitlines() if line.startswith(b'{"status":')];assert len(messages)==1 and messages[0]['proof_sha256']==p['parent_proof_sha256']
        result={'status':'separate_Z128_constructor_SDK_static_runtime_origin_replayed' if passed else 'separate_Z128_constructor_runtime_observation_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'CPU_factory_contracts_reused':1400,'runtime_identity_established_for_new_observation':passed,'phases':phases,'source_runtime_SDK_unchanged':True,'old_wrong_named_CUDART_DLL_field_remains_quarantined':True,'actual_Garnet_plugin_rewired':False,'GPU_model_or_matmul_invoked':False,'numerical_equivalence_established':False,'full_model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes())}
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'SDK_outcomes':{name:row['exit_code'] for name,row in phases.items() if name.startswith('trace-')}}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
