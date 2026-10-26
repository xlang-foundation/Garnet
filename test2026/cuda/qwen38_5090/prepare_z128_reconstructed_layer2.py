"""Prepare real layer2 correctness only after independently replayed actual native build."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,zipfile
P='Garnet/test2026/cuda/qwen38_5090/';PARENT='z128-reconstructed-plugin-build-attempt2';sha=lambda b:hashlib.sha256(b).hexdigest()
NAMES=('z128_reconstructed_layer2_layout.py','z128_reconstructed_layer2_replay.py','z128_reconstructed_layer2_trace.py','test_z128_reconstructed_layer2.py','probe_z128_reconstructed_layer2.py','test_z128_reconstructed_layer2.bat','run_z128_reconstructed_layer2.py','prepare_z128_reconstructed_layer2.py','verify_z128_reconstructed_layer2.py','evidence/z128-reconstructed-layer2-derivation.json','evidence/z128-reconstructed-layer2-owner-derivation.json','evidence/layer2-ordinary-common-input.f32','evidence/layer2-ordinary-common-input.json')
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(m),str(repo)]
    proof=(args.proofs/(PARENT+'-verified.json')).read_bytes();assert proof==(args.proofs/(PARENT+'-remote-verified.json')).read_bytes();q=json.loads(proof);assert q['status']=='reconstructed_Z128_plugin_native_CPU_build_replayed' and q['exit_code']==0 and q['source_runtime_SDK_checkpoint_CMake_restored']
    archive=args.proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'))
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        prepname=P+'evidence/z128-reconstructed-plugin-build-preparation-attempt2.json';prep=z.read(prepname);bp=json.loads(prep);w=json.loads(z.read('results/'+PARENT+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['candidate_library_sha256']==q['candidate_library_sha256']
        source=z.read(bp['bundle']);assert sha(source)==bp['bundle_sha256'] and len(bp['candidate_files'])==43
    target=m/'evidence/z128-reconstructed-plugin-build-preparation-attempt2.json';assert not target.exists() or target.read_bytes()==prep;target.write_bytes(prep)
    parents={PARENT:dict(status=q['status'],exit_code=0,proof_sha256=sha(proof),archive_sha256=q['archive_sha256'])}
    old='layer2-ordinary-witness-attempt1';oldproof=(args.proofs/(old+'-verified.json')).read_bytes();oldq=json.loads(oldproof);assert oldq['status']=='layer2_ordinary_failure_preserved' and sha((args.proofs/(old+'-evidence.zip')).read_bytes())==oldq['archive_sha256'];parents[old]=dict(status=oldq['status'],exit_code=1,proof_sha256=sha(oldproof),archive_sha256=oldq['archive_sha256'])
    lp=json.loads((m/'evidence/layer2-ordinary-witness-preparation.json').read_bytes());inherited={**bp['base_files'],**bp['methods'],prepname:sha(prep)};new={P+n:sha((m/n).read_bytes()) for n in NAMES};files={**inherited,**new}
    assert not any('probe_native_external_prefix_batch.py' in n for n in files)
    for n in NAMES:
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    tests=['test2026.cuda.qwen38_5090.test_z128_reconstructed_plugin','test2026.cuda.qwen38_5090.test_z128_reconstructed_plugin_retry','test2026.cuda.qwen38_5090.test_z128_reconstructed_layer2']
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit real layer2 correctness methods and qualified new plugin; unrelated probe excluded','owned_source_sha256':{**bp['methods'],**new}};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*tests],args.work/'z128-reconstructed-layer2-local-cpu-ledger','correctness','12CPU new composition/collector/fault checks including86historical controls; every rehashed hidden/history/state difference refused, no local GPU',[m/n for n in NAMES])==0
    p={'status':'z128_reconstructed_layer2_prepared_gpu_unrun','files':files,'new_files':new,'inherited_files':inherited,'candidate_files':bp['candidate_files'],'candidate_base':bp['candidate_base'],'installed_files':{**files,**{'Garnet/'+n:h for n,h in bp['candidate_files'].items()}},'runtime':bp['runtime'],'environment':bp['environment'],'checkpoints':bp['checkpoints'],'parents':parents,'candidate_library_sha256':q['candidate_library_sha256'],'source_bundle':bp['bundle'],'source_bundle_sha256':bp['bundle_sha256'],'audit_dependencies':lp['audit_dependencies'],'test_ids':tests,'expected_cpu_test_count':12,'gpu_uuid':lp['gpu_uuid'],'common_input_sha256':lp['common_input_sha256'],'common_input_metadata':lp['common_input_metadata'],'scope':'Actual reconstructed Z128 Garnet plugin with qualified SDK-built factory. Strict optin1+canonical1 selects new1/old0 only T128; T32 both0. Same real128 layer2 input, all73finite arrays/readonly/carried state/repeats, original58T32entry byte-parity. Save exact arithmetic differences then require all655360hiddenF32 words and finalhistory/state bit-exact before traceoff or further admission. Two fresh native processes traceon/unset; compare all73saved arrays. Separate-TU old observer removed. No changed math/tolerance/oracle/full4096 model or performance claim. Restore14old/29new targets/library/runtime/SDK/checkpoint and retire only owned cachefiles.'}
    prepfile=m/'evidence/z128-reconstructed-layer2-preparation.json';assert not prepfile.exists();raw=json.dumps(p,indent=2).encode();prepfile.write_bytes(raw);deploy={n:(repo/n.removeprefix('Garnet/')).read_bytes() for n in new};deploy[P+'evidence/'+prepfile.name]=raw
    package=args.work/'z128-reconstructed-layer2-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in deploy.items():z.writestr(n,b)
        z.writestr('install-manifest.json',json.dumps({'files':{n:sha(b) for n,b in deploy.items()}},indent=2))
    result={'status':p['status'],'package_sha256':sha(package.read_bytes()),'package_bytes':package.stat().st_size,'source_pins':len(files),'candidate_files':43,'preparation_sha256':sha(raw)};(args.work/'z128-reconstructed-layer2-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
