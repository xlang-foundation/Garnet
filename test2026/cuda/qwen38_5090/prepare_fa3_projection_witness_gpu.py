"""Prepare an actual diagnostic observation after paired native build and real input proofs."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,zipfile
P='Garnet/test2026/cuda/qwen38_5090/';PARENT='fa3-projection-witness-build-attempt1';INPUT='z128-reconstructed-layer2-attempt1';sha=lambda b:hashlib.sha256(b).hexdigest()
NAMES=('fa3_projection_witness.py','test_fa3_projection_witness.py','probe_fa3_projection_witness.py','test_fa3_projection_witness.bat','run_fa3_projection_witness.py','prepare_fa3_projection_witness_gpu.py','verify_fa3_projection_witness.py','evidence/fa3-projection-witness-owner-derivation.json','evidence/reconstructed-fa3-common-input.f32','evidence/reconstructed-fa3-common-input.json')
def main():
    a=argparse.ArgumentParser();a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args();args.proofs=args.proofs.resolve();args.work=args.work.resolve();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(m),str(repo)]
    parents={}
    for job in (PARENT,INPUT):
        proof=(args.proofs/(job+'-verified.json')).read_bytes();assert proof==(args.proofs/(job+'-remote-verified.json')).read_bytes();q=json.loads(proof);assert sha((args.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256'];parents[job]=dict(status=q['status'],exit_code=0,proof_sha256=sha(proof),archive_sha256=q['archive_sha256'])
        if job==PARENT:assert q['status']=='FA3_projection_witness_native_CPU_build_replayed' and q['exit_code']==0 and q['source_runtime_SDK_checkpoint_CMake_restored'];build=q
        else:assert q['layer2_cross_profile_admitted'] and q['source_runtime_SDK_checkpoint_restored']
    with zipfile.ZipFile(args.proofs/(PARENT+'-evidence.zip')) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'))
        for n,h in manifest['files'].items():assert sha(z.read(n))==h
        prepname=P+'evidence/fa3-projection-witness-build-preparation.json';raw=z.read(prepname);bp=json.loads(raw);w=json.loads(z.read('results/'+PARENT+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['candidate_library_sha256']==build['candidate_library_sha256']
    target=m/'evidence/fa3-projection-witness-build-preparation.json';assert not target.exists() or target.read_bytes()==raw;target.write_bytes(raw)
    inherited={**bp['base_files'],**bp['methods'],prepname:sha(raw)};new={P+n:sha((m/n).read_bytes()) for n in NAMES};files={**inherited,**new};assert not any('probe_native_external_prefix_batch.py' in n for n in files)
    metadata=json.loads((m/'evidence/reconstructed-fa3-common-input.json').read_bytes());fixture=(m/'evidence/reconstructed-fa3-common-input.f32').read_bytes();assert len(fixture)==2621440 and sha(fixture)==metadata['sha256'] and metadata['paired_source_proof_sha256']==parents[INPUT]['proof_sha256'] and metadata['source_archive_sha256']==parents[INPUT]['archive_sha256']
    for n in NAMES:
        if n.endswith('.py'):ast.parse((m/n).read_bytes())
    tests=['test2026.cuda.qwen38_5090.test_z128_reconstructed_plugin','test2026.cuda.qwen38_5090.test_z128_reconstructed_plugin_retry','test2026.cuda.qwen38_5090.test_fa3_projection_witness_source','test2026.cuda.qwen38_5090.test_fa3_projection_witness']
    import run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit actual FA3 projection observation methods; unrelated untracked excluded','owned_source_sha256':{**bp['methods'],**new}};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*tests],args.work/'fa3-projection-witness-gpu-local-cpu-ledger','correctness','14source/array/readonly/repeat/metadata fault controls including86exact historical assertions; no local native or GPU',[m/n for n in NAMES])==0
    old=json.loads((m/'evidence/z128-reconstructed-layer2-preparation.json').read_bytes());p=dict(status='FA3_projection_witness_prepared_gpu_unrun',files=files,new_files=new,inherited_files=inherited,candidate_files=bp['candidate_files'],candidate_base=bp['candidate_base'],installed_files={**files,**{'Garnet/'+n:h for n,h in bp['candidate_files'].items()}},runtime=bp['runtime'],environment=bp['environment'],checkpoints=bp['checkpoints'],parents=parents,candidate_library_sha256=build['candidate_library_sha256'],source_bundle=bp['bundle'],source_bundle_sha256=bp['bundle_sha256'],audit_dependencies=[],test_ids=tests,expected_cpu_test_count=14,gpu_uuid=old['gpu_uuid'],common_input_sha256=metadata['sha256'],scope='Actual Garnet explicit layer3 normalization/K/V observation on the admitted real layer2 common128 input. Same native primitives and embedded FP8 K/V weights, exact B1 T32/T128, three warmups/two repeats, all51finite entries and3670016outputF32 words, input readonly/repeat guards. Differences are retained diagnostic evidence, not numerical promotion or performance. Observe K/V before a redundant unchanged allGDN rerun to target the unresolved boundary; original509GDN/all4K/quality/concurrency gates remain mandatory before promotion. Restore14old+30new targets, native/runtime/SDK/checkpoints, retire only owned cache. No benchmark/2x claim.')
    target=m/'evidence/fa3-projection-witness-gpu-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw);deploy={n:(repo/n.removeprefix('Garnet/')).read_bytes() for n in new};deploy[P+'evidence/'+target.name]=raw;package=args.work/'fa3-projection-witness-gpu-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for n,b in deploy.items():z.writestr(n,b)
        z.writestr('install-manifest.json',json.dumps({'files':{n:sha(b) for n,b in deploy.items()}},indent=2))
    result=dict(status=p['status'],package_sha256=sha(package.read_bytes()),package_bytes=package.stat().st_size,source_pins=len(files),candidate_files=44,preparation_sha256=sha(raw));(args.work/'fa3-projection-witness-gpu-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
