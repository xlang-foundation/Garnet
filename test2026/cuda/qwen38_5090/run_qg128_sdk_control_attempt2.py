"""Exclusive standalone real SDK geometry check; production runtime/source stay unchanged."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='qg128-sdk-control-attempt2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/qg128-sdk-control-preparation-attempt2.json';p=json.loads(prep.read_bytes());baseline=json.loads((M/'evidence/full-chunk-stage-vendor-preparation.json').read_bytes())
    for n,h in p['files'].items():assert sha(ROOT/n)==h
    for n,h in {**baseline['files'],**baseline['checkpoints']}.items():assert sha(ROOT/n)==h
    for n,h in baseline['runtime'].items():assert sha(ROOT/'build/bin'/n)==h
    for n,h in baseline['environment'].items():assert sha(Path(n))==h
    peer=ROOT/'results/full-chunk-stage-vendor-optional-fields-verified.json';q=json.loads(peer.read_bytes());assert sha(peer)==p['paired_vendor_recovery_proof_sha256'] and q['status']=='actual_complete_logged_stages_recovered_CPU_optional_vendor_field_reader_only'
    original=ROOT/'results/full-chunk-stage-vendor-attempt1-verified.json';assert sha(original)==q['original_failure_proof_sha256'];assert sha(ROOT/'results/full-chunk-stage-vendor-attempt1-evidence.zip')==q['original_GPU_archive_sha256']
    previous=ROOT/'results/qg128-sdk-control-attempt1-verified.json';previousq=json.loads(previous.read_bytes());assert sha(previous)==p['previous_SDK_failure_proof_sha256'] and previousq['exit_code']==1 and previousq['source_library_runtime_SDK_checkpoint_unchanged'];assert sha(ROOT/'results/qg128-sdk-control-attempt1-evidence.zip')==previousq['archive_sha256'];driver=Path(os.environ['SystemRoot'])/'System32/nvcuda.dll';driver_before=sha(driver)
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();os.chdir(ROOT/'Garnet');os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['PATH']=str(ROOT/'build/bin')+os.pathsep+str(ROOT/'tools/cuda13/nvidia/cu13/bin/x86_64')+os.pathsep+os.environ['PATH']
    for n in ('CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_MASK','CUBLASLT_LOG_FILE'):os.environ.pop(n,None)
    ledger.source_identity=lambda:{'scope':'Explicit owned standalone QG SDK metadata source; original Garnet/native/model source unchanged and protected unrelated untracked excluded','owned_source_sha256':{n:sha(ROOT/n) for n in p['files']}};common=[ROOT/n for n in p['files']]+[prep,peer,original,previous];code=1;w=dict(status='running',preparation_sha256=sha(prep),source_before=baseline['files'],runtime_before=baseline['runtime'],SDK_before=baseline['environment'],checkpoint_before=baseline['checkpoints'],vendor_recovery_proof_sha256=sha(peer),no_model_weights_or_matmul=True,cleanup_finished=False,previous_SDK_failure_proof_sha256=sha(previous),driver_module_sha256_before=driver_before)
    try:
        dumpbin=Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/dumpbin.exe');oldexe=ROOT/'results/qg128-sdk-control-attempt1/native-build/qg128-sdk-control.exe'
        code=ledger.run([str(dumpbin),'/DEPENDENTS',str(oldexe)],out/'dependency-ledger','correctness','Record actual first-attempt PE imports; optional cudart module absence is an observed deployment fact, no link-mode assumption',common+[dumpbin,oldexe])
        if code:raise RuntimeError('PE dependency inspection failed')
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_qg128_sdk_control_attempt2.bat'),str(out/'native-build')],out/'build-ledger','build','Actual NVCC/MSVC /MT standalone QG descriptor probe, shared/hybrid link request with actual module audit, no production/native build change',common)
        if code:raise RuntimeError('SDK probe compilation failed')
        exe=out/'native-build/qg128-sdk-control.exe';w['executable_sha256']=sha(exe);idle()
        code=ledger.run([str(exe),str(out/'report.json')],out/'SDK-ledger','correctness','Actual pinned SDK first-success M16 QG control matches observed T32 public configuration; same SDK object AlgoCheck M16/M32/M128, no model matmul',common+[exe]);idle()
        if code:raise RuntimeError('Actual SDK compatibility probe failed')
        r=json.loads((out/'report.json').read_bytes());assert r['status']=='QG_M16_SDK_control_M16_M32_M128_compatible'
        w['loaded_module_sha256']={n:sha(Path(path)) for n,path in r['loaded_modules'].items()};assert w['loaded_module_sha256']['cublasLt64_13.dll']==baseline['runtime']['cublasLt64_13.dll'] and Path(r['loaded_modules']['nvcuda.dll']).resolve()==driver.resolve() and w['loaded_module_sha256']['nvcuda.dll']==driver_before;optional=r['optional_cudart_module_path'];w['optional_cudart_module_sha256']=sha(Path(optional)) if optional else None;assert not optional or w['optional_cudart_module_sha256']==baseline['runtime']['cudart64_13.dll'];w['status']='actual_QG_SDK_control_checked_needs_independent_replay'
    except BaseException as e:w.update(status='failed',error=repr(e));code=code or 1
    finally:
        idle();w.update(exit_code=code,source_after={n:sha(ROOT/n) for n in baseline['files']},runtime_after={n:sha(ROOT/'build/bin'/n) for n in baseline['runtime']},SDK_after={n:sha(Path(n)) for n in baseline['environment']},checkpoint_after={n:sha(ROOT/n) for n in baseline['checkpoints']},cleanup_finished=True,driver_module_sha256_after=sha(driver));assert w['driver_module_sha256_before']==w['driver_module_sha256_after'];assert w['source_before']==w['source_after'] and w['runtime_before']==w['runtime_after'] and w['SDK_before']==w['SDK_after'] and w['checkpoint_before']==w['checkpoint_after'];(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
        names={n.relative_to(ROOT).as_posix():n for n in [*[f for f in out.rglob('*') if f.is_file()],*common]}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n,path in names.items():z.write(path,n)
            z.writestr('evidence-manifest.json',json.dumps(dict(files={n:sha(path) for n,path in names.items()},linked_vendor_archive_sha256=q['original_GPU_archive_sha256'],linked_failed_SDK_archive_sha256=previousq['archive_sha256']),indent=2))
        print(json.dumps(dict(status=w['status'],archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files=len(names))))
    return code
if __name__=='__main__':raise SystemExit(main())
