"""Prepare native ABI cold/post-failure audit from closed observer and CPU adapter evidence."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest();OBSERVER='fp8-z128-context-audit-attempt1';CPU='fp8-z128-native-package-import-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/'
NAMES=('fp8_z128_native_audit_bridge.py','fp8_z128_native_audit_contract.py','test_fp8_z128_native_package_audit.py','prepare_fp8_z128_native_package_audit.py','verify_fp8_z128_native_package_audit.py','run_fp8_z128_native_package_audit.py','probe_fp8_z128_native_audit.py','probe_fp8_z128_native_audit_cold.py','probe_fp8_z128_native_audit_abi.py','test_fp8_z128_native_package_audit.bat','test_fp8_z128_native_audit_cold.bat','test_fp8_z128_native_audit_abi.bat','build_fp8_z128_context_native_eh.bat','fp8_z128_context_native_observed.cpp','evidence/fp8-z128-native-package-audit-derivation.json','analyze_fp8_z128_native_package_audit.py','evidence/fp8-z128-native-package-audit-analyzer-derivation.json')

NAMES+=('evidence/fp8-z128-native-package-audit-verifier-derivation.json',)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    proof=(args.proofs/(OBSERVER+'-verified.json')).read_bytes();q=json.loads(proof);assert proof==(args.proofs/(OBSERVER+'-remote-verified.json')).read_bytes() and q['status']=='fp8_z128_observer_compiled_but_embedded_libffi_call_failure_preserved' and not q['main_graph_sequence_started']
    archive=args.proofs/(OBSERVER+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;praw=z.read(PREFIX+'evidence/fp8-z128-context-audit-preparation.json');p=json.loads(praw);w=json.loads(z.read('results/'+OBSERVER+'/workflow.json'))
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before_sha256']==w['source_after_sha256']==p['files']
        observer='results/'+OBSERVER+'/build/context-audit.dll';assert sha(z.read(observer))==q['observer_library_sha256']
    cpu=(args.proofs/(CPU+'-verified.json')).read_bytes();cq=json.loads(cpu);assert cpu==(args.proofs/(CPU+'-remote-verified.json')).read_bytes() and cq['status']=='native_XLang_package_CPU_invocation_and_unchanged_SDK_replayed' and cq['workflow_exit_code']==0 and cq['report']['abi']==24 and cq['report']['cpu_native_calls']==4
    assert sha((args.proofs/(CPU+'-evidence.zip')).read_bytes())==cq['archive_sha256']
    with zipfile.ZipFile(args.proofs/(CPU+'-evidence.zip')) as z:
        assert z.testzip() is None;cw=json.loads(z.read('results/'+CPU+'/workflow.json'));assert cw['before']==cw['after'] and cw['cleanup_finished'] and cw['exit_code']==0
        assert all(cw['before']['source'][name]==h for name,h in p['files'].items());assert cw['before']['runtime']==p['runtime'] and cw['before']['environment']==p['environment'];p['files']=cw['before']['source']
        assert sha((m/'fp8_z128_context_native.cpp').read_bytes())==p['files'][PREFIX+'fp8_z128_context_native.cpp']
    p['files'][PREFIX+'evidence/fp8-z128-context-audit-preparation.json']=sha(praw);p['files'][observer]=q['observer_library_sha256'];p['observer_library']=observer;p['observer_library_sha256']=q['observer_library_sha256']
    new={PREFIX+name:sha((m/name).read_bytes()) for name in NAMES}
    for name in NAMES:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    for name,h in p['candidate_files'].items():assert sha((repo/name).read_bytes())==h
    p['files'].update(new);p['new_files']=new;p['installed_files']={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
    def ids(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from ids(item)
            else:yield item.id()
    p['test_ids']=sorted(p['test_ids']+list(ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_fp8_z128_native_package_audit'))));assert len(p['test_ids'])==len(set(p['test_ids']))==42
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit native adapter and candidate source pins; unrelated untracked excluded','owned_source_sha256':{**new,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*p['test_ids']],args.work/'fp8-z128-native-package-audit-local-cpu-ledger','correctness','42 retained/native ABI receipt/provenance/control-isolation fault checks; synthetic cases not GPU or benchmarks; no local native/GPU',[m/name for name in NAMES])==0
    p['parents'][OBSERVER]={'status':q['status'],'exit_code':1,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']};p['parents'][CPU]={'status':cq['status'],'exit_code':0,'proof_sha256':sha(cpu),'archive_sha256':cq['archive_sha256']}
    p.update(status='fp8_z128_native_package_audit_prepared_gpu_unrun',expected_cpu_test_count=42,scope='Existing native package ABI24, EHsc adapter plus readonly loaded-module-path identity. CPU ABI/path/hash guards first, separatecold constructor then same closednative28 Garnet DLL/frozen main real128collector/order with nativecall only after firstT128buildfailure. Reuse exact observedhelperDLL from closedlibffi failure, no ctypes/libffi/SDK/productionDLL source change. RealAPIwrappers return unchanged; extraIDqueries/IO after actualhelperreturns, no modelmatmuls. Same thread/process/CUDAcontext but separateobservedhelper, not directinstrumentation insideGarnetplugin/fullmodelcause. Restore8old/20new/nativebaseline/runtime/SDK/checkpoints/ownedcache;42CPUcontrols, completecandidateopaque and partialT32 replay; no model/quality/performance/2xadmission.')
    target=m/'evidence/fp8-z128-native-package-audit-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    payload={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};payload[PREFIX+'evidence/'+target.name]=raw;source=args.work/'fp8-z128-native-source.zip';assert sha(source.read_bytes())==p['source_bundle_sha256'];payload[p['source_bundle']]=source.read_bytes();package=args.work/'fp8-z128-native-package-audit-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in payload.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in payload.items()}},indent=2))
    result={'status':p['status'],'files':len(p['files']),'installed':len(p['installed_files']),'targets':len(payload),'CPU_tests':42,'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw),'observer_library_sha256':p['observer_library_sha256']};(args.work/'fp8-z128-native-package-audit-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))

if __name__=='__main__':main()
