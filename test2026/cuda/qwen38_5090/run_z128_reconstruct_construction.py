"""Isolated original/new helper CPU contracts and separate SDK constructor; no production installation."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='z128-reconstruct-construction-attempt1';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/z128-reconstruct-construction-preparation.json';p=json.loads(prep.read_bytes());pins={**p['files'],**p['methods']}
    def snapshot():return {name:sha((ROOT/name).read_bytes()) for name in pins}
    assert p['status']=='separate_Z128_reconstruction_constructor_prepared_unrun' and snapshot()==pins
    for name,h in p['runtime'].items():assert sha((ROOT/'build/bin'/name).read_bytes())==h
    for name,h in p['environment'].items():assert sha(Path(name).read_bytes())==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');assert sha(proof.read_bytes())==row['proof_sha256'] and json.loads(proof.read_bytes())['status']==row['status'] and sha((ROOT/'results'/(job+'-evidence.zip')).read_bytes())==row['archive_sha256']
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();os.chdir(ROOT/'Garnet');code=1
    w={'status':'running','preparation_sha256':sha(prep.read_bytes()),'source_before':snapshot(),'runtime_before':p['runtime'],'environment_before':p['environment'],'production_source_or_library_modified':False,'performance_admitted':False,'GPU_model_or_matmul_invoked':False,'numerical_equivalence_established':False}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit owned separate constructor inputs; unrelated untracked excluded','owned_source_sha256':snapshot()};common=[prep]+[ROOT/name for name in p['methods']]+[ROOT/'results'/(job+'-verified.json') for job in p['parents']];save()
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_z128_reconstruct'],out/'cpu-source-ledger','correctness','Threefocused exact source/legacy/default/configuration provenance and fault-envelope controls; no model source installation',common)
        if code:raise RuntimeError('Focused CPU source controls')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_z128_reconstruct_cpu.bat'),p['vcvars'],str(out/'cpu-build')],out/'cpu-build-ledger','build','CPU fixture actual helper CPP, no CUDA/Garnet/SDK linkage; supported and unsupportedCUDART variants',common)
        if code:raise RuntimeError('Actual helper CPU build')
        code=ledger.run([sys.executable,str(M/'run_z128_reconstruct_cpu.py'),'--build',str(out/'cpu-build'),'--parent',str(ROOT/'results/z128-factory-trace-native-build-attempt1-evidence.zip'),'--output',str(out/'cpu')],out/'cpu-contract-ledger','correctness','1400CPU processes:480retained legacy transcripts+910reconstruction failures/exceptions/configdrift/cache+10unsupported; trace policies do not change SDK transcript',common)
        if code:raise RuntimeError('Actual helper CPU contracts')
        for job,verifier,expected in (('z128-factory-layer2-attempt1','verify_z128_factory_layer2.py',1),('z128-opaque-config-audit-attempt2','verify_z128_opaque_config_audit_attempt2.py',0)):
            code=ledger.run([sys.executable,str(M/verifier),'--proofs',str(ROOT/'results')],out/(job+'-replay-ledger'),'correctness','Replay retained complete parent evidence, not a model rerun. Expected scientific failure retained where applicable',common)
            rows=[json.loads(line) for line in (out/(job+'-replay-ledger/commands.log')).read_bytes().splitlines() if line.startswith(b'{"status":')];assert code==expected and len(rows)==1 and rows[0]['status']==p['parents'][job]['status'] and rows[0]['proof_sha256']==p['parents'][job]['proof_sha256'];code=0
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_z128_reconstruct_sdk.bat'),p['vcvars'],str(out/'sdk-build')],out/'sdk-build-ledger','build','Build same explicit separate constructor CPP against pinned real CUDA/cuBLAS headers/import libraries; production Garnet unchanged',common)
        if code:raise RuntimeError('RealSDK constructor build')
        for phase,policy in (('trace-on','1'),('trace-unset','0')):
            code=ledger.run([sys.executable,str(M/'monitor_native_probe.py'),'--study',str(ROOT),'--output',str(out/(phase+'-resources')),'--',sys.executable,str(M/'run_z128_reconstruct_sdk_process.py'),'--root',str(ROOT),'--executable',str(out/'sdk-build/constructor.exe'),'--output',str(out/phase),'--policy',policy],out/(phase+'-ledger'),'correctness','Oneexclusive actualSDK constructor/cache/configuration/M32+M128AlgoCheck observation; no model/matmul/timing qualification',common+[out/'sdk-build/constructor.exe'])
            idle()
            if code:raise RuntimeError('ActualSDK separate constructor failed; preserve trace')
        w.update(status='separate_API_reconstruction_constructor_CPU_and_SDK_collected',SDK_constructor_executable_sha256=sha((out/'sdk-build/constructor.exe').read_bytes()))
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
