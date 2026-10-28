"""Read public configuration attributes of retained descriptors; no production writes or GPU matmul."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='z128-opaque-config-audit-attempt2';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    from z128_opaque_config import replay
    prep=M/'evidence/z128-opaque-config-preparation-attempt2.json';p=json.loads(prep.read_bytes());pins={**p['files'],**p['methods']}
    def snapshot():return {name:sha((ROOT/name).read_bytes()) for name in pins}
    assert snapshot()==pins
    for name,h in p['runtime'].items():assert sha((ROOT/'build/bin'/name).read_bytes())==h
    for name,h in p['environment'].items():assert sha(Path(name).read_bytes())==h
    parent=ROOT/'results'/p['parent_job'];assert sha(parent.with_name(p['parent_job']+'-verified.json').read_bytes())==p['parent_proof_sha256'] and sha(parent.with_name(p['parent_job']+'-evidence.zip').read_bytes())==p['parent_archive_sha256']
    q=json.loads(parent.with_name(p['parent_job']+'-verified.json').read_bytes());assert q['actual_constructor_rejection_guard_observed'] and q['actual_constructor_rejection_guards'][0]['heuristic_candidates'][0]['opaque']==p['actual']
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();os.chdir(ROOT/'Garnet');code=1
    w={'status':'running','preparation_sha256':sha(prep.read_bytes()),'source_before':snapshot(),'production_source_or_library_modified':False,'GPU_model_or_matmul_executed':False,'performance_admitted':False}
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    ledger.source_identity=lambda:{'scope':'Explicit public attribute audit inputs; unrelated untracked excluded','owned_source_sha256':snapshot()};common=[prep,parent.with_name(p['parent_job']+'-verified.json')]+[ROOT/name for name in p['methods']];save()
    try:
        previous=p['previous_failure'];previous_proof=ROOT/'results'/(previous['job']+'-verified.json');previous_archive=ROOT/'results'/(previous['job']+'-evidence.zip')
        assert sha(previous_proof.read_bytes())==previous['proof_sha256'] and sha(previous_archive.read_bytes())==previous['archive_sha256'];common.append(previous_proof)
        code=ledger.run([sys.executable,str(M/'verify_z128_factory_layer2.py'),'--proofs',str(ROOT/'results')],out/'retained-parent-ledger','correctness','Replay immutable complete parent archive:88CPU installed-candidate controls, actual failed factory telemetry,58T32 arrays, restoration. Expected scientific failure exit1; no model rerun',common)
        messages=[json.loads(line) for line in (out/'retained-parent-ledger/commands.log').read_bytes().splitlines() if line.startswith(b'{"status":')];assert code==1 and len(messages)==1 and messages[0]['status']==q['status'] and messages[0]['proof_sha256']==p['parent_proof_sha256']
        code=0
        code=ledger.run([sys.executable,'-m','unittest',*p['test_ids']],out/'cpu-ledger','correctness','Query-only CPU fault control; parent88 installed-candidate controls replayed separately and unchanged',common)
        if code:raise RuntimeError('CPU configuration query controls')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_z128_opaque_config_cpu.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Compile public18attribute getter only; no context constructor/Garnet library/matmul',common)
        if code:raise RuntimeError('Public getter build')
        os.environ['PATH']=str(ROOT/'build/bin')+os.pathsep+os.environ['PATH']
        code=ledger.run([str(out/'build/config-query.exe'),p['expected'],p['actual']],out/'query-ledger','correctness','Read nine documented attributes from each exact retained opaque descriptor; no setter/AlgoCheck/context/matmul',common+[out/'build/config-query.exe'])
        if code:raise RuntimeError('Public getter execution')
        data=[json.loads(line) for line in (out/'query-ledger/commands.log').read_bytes().splitlines() if line.startswith(b'{"status":"retained_opaque_')];assert len(data)==1;report=data[0]
        loaded=Path(bytes.fromhex(report['module_path_hex']).decode());assert loaded.resolve()==(ROOT/'build/bin/cublasLt64_13.dll').resolve() and sha(loaded.read_bytes())==p['runtime']['cublasLt64_13.dll']
        (out/'report.json').write_bytes(json.dumps(report,indent=2).encode());w.update(status='retained_opaque_documented_attributes_collected',replay=replay(report,p['expected'],p['actual']),executable_sha256=sha((out/'build/config-query.exe').read_bytes()))
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        w.update(exit_code=code,source_after=snapshot(),runtime_after={name:sha((ROOT/'build/bin'/name).read_bytes()) for name in p['runtime']},environment_after={name:sha(Path(name).read_bytes()) for name in p['environment']},cleanup_finished=True);save()
        paths=[path for path in out.rglob('*') if path.is_file()]+common;names={path.relative_to(ROOT).as_posix():path for path in paths}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path.read_bytes()) for name,path in names.items()}},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':len(names)}))
    return code
if __name__=='__main__':raise SystemExit(main())
