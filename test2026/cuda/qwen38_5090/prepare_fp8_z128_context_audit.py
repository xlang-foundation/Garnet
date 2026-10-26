"""CPU preparation for actual-helper observer and same-DLL post-failure diagnostic."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest();PARENT='fp8-z128-stage-trace-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/'
NAMES=('fp8_z128_context_audit.cpp','build_fp8_z128_context_audit.bat','fp8_z128_context_audit.py','fp8_z128_audit_bridge.py','probe_fp8_z128_audit_cold.py','test_fp8_z128_audit_cold.bat','probe_fp8_z128_context_audit.py','test_fp8_z128_context_audit.bat','run_fp8_z128_context_audit.py','prepare_fp8_z128_context_audit.py','verify_fp8_z128_context_audit.py','test_fp8_z128_context_audit.py','evidence/fp8-z128-context-audit-derivation.json','probe_fp8_z128_ctypes.py','test_fp8_z128_ctypes.bat','probe_fp8_z128_ctypes_bound.py','test_fp8_z128_ctypes_bound.bat','evidence/fp8-z128-ctypes-bound-derivation.json','probe_fp8_z128_ctypes_bound2.py','test_fp8_z128_ctypes_bound2.bat','evidence/fp8-z128-ctypes-bound2-derivation.json')

NAMES+=('analyze_fp8_z128_context_audit.py','evidence/fp8-z128-context-audit-analyzer-derivation.json','analyze_fp8_z128_stage_trace_failure.py')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    proof=(args.proofs/(PARENT+'-verified.json')).read_bytes();q=json.loads(proof);assert proof==(args.proofs/(PARENT+'-remote-verified.json')).read_bytes() and q['status']=='fp8_z128_same_DLL_initialization_trace_preserved' and q['workflow_exit_code']==1
    archive=args.proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;praw=z.read(PREFIX+'evidence/fp8-z128-stage-trace-preparation.json');p=json.loads(praw);w=json.loads(z.read('results/'+PARENT+'/workflow.json'))
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before_sha256']==w['source_after_sha256']==p['files']
    p['files'][PREFIX+'evidence/fp8-z128-stage-trace-preparation.json']=sha(praw)
    old=json.loads((m/'evidence/fp8-z128-native-control-preparation.json').read_bytes());helper={}
    for suffix in ('modelopt_fp8_z128_canonical.cpp','modelopt_fp8_z128_canonical.h'):
        name='native_fp8_z128_candidate/'+suffix;row=old['candidate_copies'][name];assert row['sha256']==p['candidate_files'][row['production_source']];helper[PREFIX+name]=row['sha256']
    p['files'].update(helper);p['observed_helper_files']=helper;p['vcvars']=old['vcvars'];p['expected_m32_opaque']=old['expected_m32_opaque']
    capability_names=('fp8-z128-ctypes-import.log','fp8-z128-ctypes-import-bound.log','fp8-z128-ctypes-import-bound2.log','fp8-z128-ctypes-import-bound2.json')
    p['capability_files']=['results/'+name for name in capability_names]
    for name in capability_names:p['files']['results/'+name]=sha((args.work/name).read_bytes())
    capability=json.loads((args.work/'fp8-z128-ctypes-import-bound2.json').read_bytes());assert capability['status']=='embedded_ctypes_CPU_import_passed' and capability['pointer_bytes']==8 and capability['runtime_host_sha256']==p['runtime']['xlang3.exe'] and capability['baseline_library_sha256']==p['runtime']['garnet.dll'] and not capability['gpu_library_explicitly_loaded'] and not capability['garnet_explicitly_imported_by_probe']
    if capability['extension_module']:p['environment'][capability['extension_module']]=capability['extension_sha256']
    new={PREFIX+name:sha((m/name).read_bytes()) for name in NAMES}
    for name in NAMES:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    for name,h in p['candidate_files'].items():assert sha((repo/name).read_bytes())==h
    p['files'].update(new);p['new_files']=new;p['installed_files']={**p['files'],**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}
    def ids(suite):
        for item in suite:
            if isinstance(item,unittest.TestSuite):yield from ids(item)
            else:yield item.id()
    p['test_ids']=sorted(p['test_ids']+list(ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_fp8_z128_context_audit'))));assert len(p['test_ids'])==len(set(p['test_ids']))==37
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit observer and actual helper source pins; unrelated untracked excluded','owned_source_sha256':{**new,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}}};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*p['test_ids']],args.work/'fp8-z128-context-audit-local-cpu-ledger','correctness','37 retained and observer complete-candidate/selection-boundary/source-isolation fault controls; synthetic values not hardware evidence; no local native/GPU',[m/name for name in NAMES])==0
    p['parents'][PARENT]={'status':q['status'],'exit_code':1,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}
    p.update(status='fp8_z128_context_audit_prepared_gpu_unrun',expected_cpu_test_count=37,scope='Same closednative28 Garnet DLL and first128collector/order; no API logger. Build separate observerDLL including exactactualtypedhelperCPP+header. Realheuristic/config/check/matmul wrappers call originals once and return unchanged; record completeopaque candidates afterselection. Separatecoldprocess constructor, thenonlypostfailure observer in sameprocess/CUDAcontext afterreproduced firstmainT128buildfailure/twoT32profiles beforecleanup. AdditionalIDqueries/IO afterhelperreturn. Noalgorithmsubstitution/modelmatmul/performanceadmission. Observehelper guard separately; not directhookinsideGarnetplugin/fullmodelcausality. Restore8old/20new/source/runtime/SDK/checkpoint/ownedcache;37CPUcontrols and embeddedctypes probe failures retained.')
    target=m/'evidence/fp8-z128-context-audit-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    payload={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};payload[PREFIX+'evidence/'+target.name]=raw;source=args.work/'fp8-z128-native-source.zip';assert sha(source.read_bytes())==p['source_bundle_sha256'];payload[p['source_bundle']]=source.read_bytes()
    package=args.work/'fp8-z128-context-audit-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in payload.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in payload.items()}},indent=2))
    result={'status':p['status'],'files':len(p['files']),'installed':len(p['installed_files']),'targets':len(payload),'CPU_tests':37,'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)};(args.work/'fp8-z128-context-audit-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))

if __name__=='__main__':main()
