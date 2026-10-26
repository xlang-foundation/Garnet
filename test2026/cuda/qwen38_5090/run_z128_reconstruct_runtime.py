"""Correct SDK runtime-origin observation; reuse fully replayed unchanged CPU factory contracts."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='z128-reconstruct-runtime-attempt1';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from z128_reconstruct_runtime import identity
    prep=M/'evidence/z128-reconstruct-runtime-preparation.json';p=json.loads(prep.read_bytes());pins={**p['files'],**p['methods']}
    def snapshot():return {name:sha((ROOT/name).read_bytes()) for name in pins}
    assert snapshot()==pins
    for name,h in p['runtime'].items():assert sha((ROOT/'build/bin'/name).read_bytes())==h
    for name,h in p['environment'].items():assert sha(Path(name).read_bytes())==h
    parent=ROOT/'results/z128-reconstruct-construction-attempt2';proof=parent.with_name(parent.name+'-quarantine-verified.json');assert sha(proof.read_bytes())==p['parent_proof_sha256'] and sha(parent.with_name(parent.name+'-evidence.zip').read_bytes())==p['parent_archive_sha256'];q=json.loads(proof.read_bytes());assert q['CPU_processes']==1400 and not q['active_CUDA_runtime_identity_established']
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();os.chdir(ROOT/'Garnet');code=1
    w={'status':'running','preparation_sha256':sha(prep.read_bytes()),'source_before':snapshot(),'runtime_before':p['runtime'],'environment_before':p['environment'],'production_source_or_library_modified':False,'GPU_model_or_matmul_invoked':False,'performance_admitted':False,'CPU_factory_contracts_reused':1400}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit SDK runtime-origin methods; unrelated untracked excluded','owned_source_sha256':snapshot()};common=[prep,proof]+[ROOT/name for name in p['methods']];save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_z128_reconstruct_runtime'],out/'cpu-source-ledger','correctness','Twofocused metadata-only derivation/null-handle/static-linkage fault controls; unchanged factory CPP/CPU driver',common)
        if code:raise RuntimeError('Runtime metadata controls')
        code=ledger.run([sys.executable,str(M/'analyze_z128_reconstruct_quarantine.py'),'--proofs',str(ROOT/'results')],out/'CPU-parent-replay-ledger','correctness','Replay all1400unchanged CPU contracts plus quarantined priorSDKobservations; old bad module field remains unqualified',common)
        lines=[json.loads(line) for line in (out/'CPU-parent-replay-ledger/commands.log').read_bytes().splitlines() if line.startswith(b'{"status":')];assert code==0 and len(lines)==1 and lines[0]['proof_sha256']==p['parent_proof_sha256']
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_z128_reconstruct_runtime_sdk.bat'),p['vcvars'],str(out/'sdk-build')],out/'sdk-build-ledger','build','Unchanged actual helper CPP; corrected API-address module identity/version probe, link map and PE import evidence',common)
        if code:raise RuntimeError('SDK runtime metadata build')
        for phase,policy in (('trace-on','1'),('trace-unset','0')):
            code=ledger.run([sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(phase+'-resources')),'--',sys.executable,str(M/'run_z128_reconstruct_sdk_process.py'),'--root',str(ROOT),'--executable',str(out/'sdk-build/constructor.exe'),'--output',str(out/phase),'--policy',policy],out/(phase+'-ledger'),'correctness','Oneexclusive source-frozen actualSDK constructor/cache/config/M32+M128checks; correct static CUDART origin, no model/matmul/timing',common+[out/'sdk-build/constructor.exe'])
            idle()
            if code:raise RuntimeError('SDK constructor/runtime probe failed')
            report=json.loads((out/phase/'report.json').read_bytes())['native_report'];r=identity(report,str(out/'sdk-build/constructor.exe'),(out/'sdk-build/dependencies.txt').read_bytes(),(out/'sdk-build/constructor.map').read_bytes());observed={}
            for row in report['module_paths']:
                path=Path(bytes.fromhex(row['path_hex']).decode());h=sha(path.read_bytes());observed[row['name']]={'path':str(path),'sha256':h}
                if row['name']=='cublasLt64_13.dll':assert h==p['runtime'][row['name']]
                elif row['name']=='cudaGetDevice':assert path.resolve()==(out/'sdk-build/constructor.exe').resolve()
                else:assert row['name']=='nvcuda.dll' and path.resolve().is_relative_to(Path(os.environ['SystemRoot']).resolve())
            (out/phase/'runtime-origin.json').write_bytes(json.dumps({'identity':r,'observed_module_files':observed,'static_CUDART_library_sha256':p['environment'][str(ROOT/'tools/cuda13/nvidia/cu13/lib/x64/cudart.lib')]},indent=2).encode())
        w.update(status='separate_reconstruction_SDK_runtime_origin_collected',SDK_executable_sha256=sha((out/'sdk-build/constructor.exe').read_bytes()))
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after=snapshot(),runtime_after={name:sha((ROOT/'build/bin'/name).read_bytes()) for name in p['runtime']},environment_after={name:sha(Path(name).read_bytes()) for name in p['environment']},cleanup_finished=True);save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common;names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path.read_bytes()) for name,path in names.items()}},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
