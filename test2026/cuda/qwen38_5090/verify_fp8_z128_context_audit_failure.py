"""Preserve compiled-observer/embedded-libffi failure; main GPU sequence never started."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
JOB='fp8-z128-context-audit-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);args=parser.parse_args();archive=args.proofs/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        base='results/'+JOB+'/';praw=z.read(PREFIX+'evidence/fp8-z128-context-audit-preparation.json');p=json.loads(praw);w=json.loads(z.read(base+'workflow.json'))
        assert sha(praw)==w['preparation_sha256'] and w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and not w['performance_admitted']
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime'] and w['environment_before']==w['environment_after']==p['environment'] and w['checkpoint_after']==p['checkpoints']
        assert w['installed_source_sha256']==p['installed_files']=={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}} and len(p['candidate_files'])==28
        for name,h in {**p['new_files'],**p['observed_helper_files']}.items():assert sha(z.read(name))==h
        for parent,row in p['parents'].items():
            raw=z.read('results/'+parent+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==manifest['linked_parent_archives'][parent]
            assert raw==(args.proofs/(parent+'-verified.json')).read_bytes() and sha((args.proofs/(parent+'-evidence.zip')).read_bytes())==row['archive_sha256']
        source=z.read(p['source_bundle']);assert sha(source)==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(source)) as bundle:
            assert bundle.testzip() is None and len(bundle.namelist())==28 and set(bundle.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(bundle.read(name))==h
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as old:
            expected={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(expected)==8 and set(old.namelist())==set(expected)
            for name,h in expected.items():assert sha(old.read(name))==h
        for name,h in p['checkpoints'].items():assert sha(z.read(name))==h
        assert b'Ran 37 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
        for phase,code in (('cpu-ledger',0),('audit-build-ledger',0),('cold-ledger',1)):
            events=[json.loads(line) for line in z.read(base+phase+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==code
            assert events[0]['source']['owned_source_sha256']==events[1]['source_after']['owned_source_sha256']==p['installed_files']
        assert w['error']=="RuntimeError('Cold context observation failed')" and not any(name.startswith(base+'native/') for name in manifest['files']) and base+'cold/context-audit.json' not in manifest['files']
        log=z.read(base+'cold-ledger/commands.log');assert b'NotImplementedError: libffi is unavailable for ctypes' in log
        resource=json.loads(z.read(base+'cold-resources/resource-monitor.json'));assert resource['child_exit_code']==1 and resource['gpu']['uuid']==p['gpu_uuid']
        receipt=json.loads(z.read(base+'audit-library.json'));assert receipt['library_sha256']==w['audit_library_sha256']==sha(z.read(base+'build/context-audit.dll')) and receipt['cublaslt_sha256']==p['runtime']['cublasLt64_13.dll']
        capability=json.loads(z.read('results/fp8-z128-ctypes-import-bound2.json'));assert capability['status']=='embedded_ctypes_CPU_import_passed' and capability['extension_origin']=='built-in' and capability['tensor_preloaded_by_host'] and not capability['garnet_preloaded_by_host']
        result={'status':'fp8_z128_observer_compiled_but_embedded_libffi_call_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'candidate_library_sha256':p['candidate_library_sha256'],'observer_library_sha256':receipt['library_sha256'],'CPU_tests':37,'observer_build_exit_code':0,'cold_process_exit_code':1,'cold_error':'NotImplementedError: libffi is unavailable for ctypes','helper_observation_report_produced':False,'main_graph_sequence_started':False,'source_runtime_sdk_checkpoint_restored':True,'full_model_cause_established':False,'model_or_quality_or_performance_qualified':False,'two_times_target_achieved':False,'verifier_sha256':sha(Path(__file__).read_bytes()),'scope':'Only build/CPU qualification and preserved embedded-FFI failure. ctypes import/attributes/dlopen capability did not prove native function call support. Main graph sequence was never started. Keep original methods/archive immutable; use existing XLang native package ABI to invoke observer without SDK/libffi installation or production Garnet DLL mutation.'}
    raw=json.dumps(result,indent=2).encode();target=args.proofs/(JOB+'-verified.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'files':len(manifest['files'])}));return 0

if __name__=='__main__':raise SystemExit(main())
