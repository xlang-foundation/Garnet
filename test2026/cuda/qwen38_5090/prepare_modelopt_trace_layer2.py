"""Prepare real GPU initialization observation only after paired native-build proof."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
M='Garnet/test2026/cuda/qwen38_5090/';PARENT='modelopt-init-trace-native-build-attempt2';sha=lambda raw:hashlib.sha256(raw).hexdigest()
NAMES=('modelopt_trace_layer2.py','test_modelopt_trace_layer2.py','probe_modelopt_trace_layer2.py','test_modelopt_trace_layer2.bat','run_modelopt_trace_layer2.py','prepare_modelopt_trace_layer2.py','verify_modelopt_trace_layer2.py','evidence/modelopt-trace-layer2-derivation.json','evidence/layer2-ordinary-common-input.f32','evidence/layer2-ordinary-common-input.json','evidence/modelopt-init-trace-native-build-preparation-attempt2.json')
def ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);a=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    proof=(a.proofs/(PARENT+'-verified.json')).read_bytes();q=json.loads(proof);assert proof==(a.proofs/(PARENT+'-remote-verified.json')).read_bytes() and q['status']=='modelopt_init_trace_native_build_cpu_replayed' and q['source_and_runtime_restored'] and not q['gpu_model_executed']
    archive=a.proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'))
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        prep_name=M+'evidence/modelopt-init-trace-native-build-preparation-attempt2.json';prep=z.read(prep_name);bp=json.loads(prep);w=json.loads(z.read('results/'+PARENT+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and w['candidate_library_sha256']==q['candidate_library_sha256']
        source=z.read(bp['bundle']);assert sha(source)==bp['bundle_sha256'] and len(bp['candidate_files'])==38 and sum(h is not None for h in bp['candidate_base'].values())==14
    assert (m/'evidence/modelopt-init-trace-native-build-preparation-attempt2.json').read_bytes()==prep
    lp=json.loads((m/'evidence/layer2-ordinary-witness-preparation.json').read_bytes());parents={**bp['parents'],PARENT:{'status':q['status'],'exit_code':0,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}}
    inherited={**bp['base_files'],**bp['methods'],prep_name:sha(prep)};new={M+name:sha((m/name).read_bytes()) for name in NAMES};files={**inherited,**new}
    assert not any('probe_native_external_prefix_batch.py' in name for name in files)
    for name,h in new.items():
        raw=(repo/name.removeprefix('Garnet/')).read_bytes();assert sha(raw)==h
        if name.endswith('.py'):ast.parse(raw)
    added=sorted(ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_modelopt_trace_layer2')));tests=bp['test_ids']+added;assert len(added)==2 and len(tests)==len(set(tests))==83
    import run_logged as ledger
    pins={**bp['methods'],**new};ledger.source_identity=lambda:{'scope':'Explicit trace observation methods and qualified private native inputs; unrelated untracked excluded','owned_source_sha256':pins};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*tests],a.work/'modelopt-trace-layer2-local-cpu-ledger','correctness','83CPU: retain81 composed native/model/cache/init controls plus2 real trace-profile and exact collection-source controls; no local native/GPU',[repo/name.removeprefix('Garnet/') for name in pins])==0
    p={'status':'modelopt_trace_layer2_prepared_gpu_unrun','files':files,'inherited_files':inherited,'new_files':new,'candidate_files':bp['candidate_files'],'candidate_base':bp['candidate_base'],'installed_files':{**files,**{'Garnet/'+name:h for name,h in bp['candidate_files'].items()}},'runtime':bp['runtime'],'environment':bp['environment'],'checkpoints':bp['checkpoints'],'parents':parents,'candidate_library_sha256':q['candidate_library_sha256'],'source_bundle':bp['bundle'],'source_bundle_sha256':bp['bundle_sha256'],'audit_dependencies':lp['audit_dependencies'],'test_ids':tests,'expected_cpu_test_count':83,'gpu_uuid':lp['gpu_uuid'],'common_input_sha256':lp['common_input_sha256'],'common_input_metadata':lp['common_input_metadata'],'native_preparation_sha256':sha(prep),'retained_T32_analysis_sha256':sha((a.proofs/'layer2-ordinary-witness-attempt1-analysis.json').read_bytes()),'scope':'Native38 qualified default-off plugin init trace, actual ordinary layer2 first128 real input from retained T32 layer1. Trace-on process first with explicit T32/T128 build boundary markers,3warmups/2repeats/full finite arrays/readonly and carried-state checks. If successful fresh trace-unset process must be silent and byte-exact. Compare retained old T32 partial58arrays. Preserve failures and any actual initialize return1 route separately from postfailure separate-TU helper. No native/math changes versus qualified d3f15 DLL, no full3968model/image/MTP/quality/timing/performance admission. Restore14old/24new source/runtime/SDK/checkpoint/cache.'}
    target=m/'evidence/modelopt-trace-layer2-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    deploy={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};deploy[M+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=source;package=a.work/'modelopt-trace-layer2-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    result={'status':p['status'],'source_pins':len(files),'installed_pins':len(p['installed_files']),'new_files':len(new),'candidate_sources':38,'CPU_tests':83,'deployment_targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)};(a.work/'modelopt-trace-layer2-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
