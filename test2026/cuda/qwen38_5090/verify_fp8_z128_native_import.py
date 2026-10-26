"""Replay native package build/import/member calls and unchanged SDK/source/runtime."""
from pathlib import Path
import argparse,hashlib,json,zipfile
JOB='fp8-z128-native-package-import-attempt1';PARENT='fp8-z128-context-audit-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
NAMES=('fp8_z128_context_native.cpp','build_fp8_z128_context_native.bat','probe_fp8_z128_native_import.py','test_fp8_z128_native_import.bat','run_fp8_z128_native_import.py')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        base='results/'+JOB+'/';w=json.loads(z.read(base+'workflow.json'));praw=z.read(PREFIX+'evidence/fp8-z128-context-audit-preparation.json');p=json.loads(praw)
        assert w['before']==w['after'] and w['cleanup_finished'] and not w['gpu_model_executed'] and not w['helper_invoked'] and sha(praw)==w['preparation_sha256']
        own={PREFIX+name:sha(z.read(PREFIX+name)) for name in NAMES};assert w['before']=={'source':{**p['files'],**own},'runtime':p['runtime'],'environment':p['environment']}
        for name in NAMES:assert sha(Path(__file__).with_name(name).read_bytes())==own[PREFIX+name]
        parent=z.read('results/'+PARENT+'-verified.json');q=json.loads(parent);assert sha(parent)==w['parent_proof_sha256']==manifest['linked_parent_archive']['proof_sha256'] and q['archive_sha256']==w['parent_archive_sha256']==manifest['linked_parent_archive']['archive_sha256']
        assert parent==(args.proofs/(PARENT+'-verified.json')).read_bytes() and sha((args.proofs/(PARENT+'-evidence.zip')).read_bytes())==q['archive_sha256'] and q['status']=='fp8_z128_observer_compiled_but_embedded_libffi_call_failure_preserved'
        build=[json.loads(line) for line in z.read(base+'build-ledger/commands.jsonl').splitlines()];assert len(build)==2 and build[0]['event']=='started' and build[1]['event']=='finished'
        assert build[0]['source']['owned_source_sha256']==build[1]['source_after']['owned_source_sha256']==w['before']['source']
        report=None;cpu_code=None
        if build[1]['exit_code']==0:
            assert sha(z.read(base+'build/fp8_z128_context_native.dll'))==w['library_sha256']
            cpu=[json.loads(line) for line in z.read(base+'cpu-ledger/commands.jsonl').splitlines()];assert len(cpu)==2 and cpu[0]['event']=='started' and cpu[1]['event']=='finished'
            assert cpu[0]['source']['owned_source_sha256']==cpu[1]['source_after']['owned_source_sha256']==w['before']['source'];cpu_code=cpu[1]['exit_code']
        if w['exit_code']==0:
            assert w['status']=='native_package_CPU_invocation_collected_needs_replay' and build[1]['exit_code']==cpu_code==0
            report=json.loads(z.read(base+'report.json'));assert report==w['report'] and report['status']=='embedded_native_package_CPU_invocation_passed' and type(report['abi']) is int and report['abi']>0 and report['cpu_native_calls']==4 and report['rejection_codes']==[20,20,20]
            assert report['observer_DLL_loaded_by_adapter'] is report['helper_invoked'] is report['gpu_model_executed'] is report['model_or_performance_qualified'] is False and report['probe_sha256']==own[PREFIX+'probe_fp8_z128_native_import.py'] and report['runtime_host_sha256']==p['runtime']['xlang3.exe']
        result={'status':'native_XLang_package_CPU_invocation_and_unchanged_SDK_replayed' if w['exit_code']==0 else 'native_XLang_package_CPU_attempt_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'source_pins':len(w['before']['source']),'workflow_exit_code':w['exit_code'],'build_exit_code':build[1]['exit_code'],'CPU_exit_code':cpu_code,'library_sha256':w.get('library_sha256'),'report':report,'error':w.get('error'),'source_runtime_SDK_unchanged':True,'helper_or_model_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':manifest['scope']}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'exit_code':w['exit_code']}));return 0

if __name__=='__main__':raise SystemExit(main())
