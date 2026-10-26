"""Exclusive private adapter build and embedded native-call CPU test, no SDK mutation."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='fp8-z128-native-package-import-attempt1';PARENT='fp8-z128-context-audit-attempt1'
NAMES=('fp8_z128_context_native.cpp','build_fp8_z128_context_native.bat','probe_fp8_z128_native_import.py','test_fp8_z128_native_import.bat','run_fp8_z128_native_import.py')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path[:0]=[str(M),str(ROOT/'Garnet')]
    import run_logged as ledger
    from run_bf16_low_batch_native_build import idle
    prep=M/'evidence/fp8-z128-context-audit-preparation.json';p=json.loads(prep.read_bytes());proof=ROOT/'results'/(PARENT+'-verified.json');q=json.loads(proof.read_bytes())
    assert q['status']=='fp8_z128_observer_compiled_but_embedded_libffi_call_failure_preserved' and sha(ROOT/'results'/(PARENT+'-evidence.zip'))==q['archive_sha256']
    source={**p['files'],**{'Garnet/test2026/cuda/qwen38_5090/'+name:sha(M/name) for name in NAMES}}
    def snapshot():return {'source':{name:sha(ROOT/name) for name in source},'runtime':{name:sha(ROOT/'build/bin'/name) for name in p['runtime']},'environment':{name:sha(Path(name)) for name in p['environment']}}
    expected={'source':source,'runtime':p['runtime'],'environment':p['environment']};assert snapshot()==expected
    out=ROOT/'results'/JOB;archive=ROOT/'results'/(JOB+'-evidence.zip');assert not out.exists() and not archive.exists();idle();out.mkdir();(out/'build').mkdir();os.chdir(ROOT/'Garnet')
    ledger.source_identity=lambda:{'scope':'Explicit private native-package CPU adapter source, unrelated untracked excluded','owned_source_sha256':{name:sha(ROOT/name) for name in source}}
    common=[prep,proof]+[M/name for name in NAMES]+[ROOT/'xlang3/sdk/xlang3/xlang3.h',ROOT/'xlang3/sdk/xlang3/charconv_compat.h'];w={'status':'running','before':expected,'parent_proof_sha256':sha(proof),'parent_archive_sha256':q['archive_sha256'],'preparation_sha256':sha(prep),'gpu_model_executed':False,'helper_invoked':False,'cleanup_finished':False};code=1
    def save():(out/'workflow.json').write_bytes(json.dumps(w,indent=2).encode())
    save()
    try:
        code=ledger.run(['cmd.exe','/d','/c',str(M/'build_fp8_z128_context_native.bat'),p['vcvars'],str(out/'build')],out/'build-ledger','build','Isolated existing XLang package ABI adapter, Windows loader only; no CUDA or production Garnet DLL build',common)
        if code:raise RuntimeError('Native adapter build failed')
        library=out/'build/fp8_z128_context_native.dll';w['library_sha256']=sha(library);save()
        code=ledger.run(['cmd.exe','/d','/c',str(M/'test_fp8_z128_native_import.bat'),str(out/'report.json')],out/'cpu-ledger','correctness','Actual embedded native ABI and3reject-before-loader path calls; no observer/helper/CUDA invocation',common+[library])
        if code:raise RuntimeError('Embedded native adapter CPU invocation failed')
        w['report']=json.loads((out/'report.json').read_bytes());assert w['report']['status']=='embedded_native_package_CPU_invocation_passed' and w['report']['rejection_codes']==[20,20,20];w['status']='native_package_CPU_invocation_collected_needs_replay'
    except BaseException as error:w.update(status='failed',error=repr(error));code=code or 1
    finally:
        idle();w.update(exit_code=code,after=snapshot(),cleanup_finished=True);save();assert w['after']==w['before']
        names={path.relative_to(ROOT).as_posix():path for path in [*out.rglob('*'),*common] if path.is_file()}
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for name,path in names.items():z.write(path,name)
            z.writestr('evidence-manifest.json',json.dumps({'files':{name:sha(path) for name,path in names.items()},'linked_parent_archive':{'job':PARENT,'archive_sha256':q['archive_sha256'],'proof_sha256':sha(proof)},'scope':'CPU-only private native package build and real embedded member calls, guards reject before GPU loader. All941parent source/runtime/selectedSDK/checkpoint metadata pins unchanged. No production source/DLL/SDK mutation, no helper/model/performance admission.'},indent=2))
        print(json.dumps({'status':w['status'],'archive_sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(names)}),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
