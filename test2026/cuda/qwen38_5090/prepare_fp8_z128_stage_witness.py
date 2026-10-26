"""CPU preparation of closed native28 stage witness; no local native/GPU execution."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
BUILD='fp8-z128-native-build-attempt1';PREVIOUS='gdn-chunk-stage-witness-attempt2';PREFIX='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
NAMES=('prepare_fp8_z128_stage_witness.py','run_fp8_z128_stage_witness.py','probe_fp8_z128_stage_witness.py','verify_fp8_z128_stage_witness.py','test_fp8_z128_stage_witness.bat','test_fp8_z128_policy_control.bat','fp8_z128_stage_layout.py','fp8_z128_stage_replay.py','test_fp8_z128_stage_witness.py','evidence/fp8-z128-stage-probe-derivation.json','evidence/fp8-z128-stage-owner-derivation.json','evidence/native-host-stage-cases.json')
MODULES=('test_gdn_chunk_stage_witness','test_gdn_chunk_stage_bound','test_gdn_z_projection','test_gdn_z_algorithm_log','test_fp8_z128_native_control','test_fp8_z128_native_source','test_fp8_z128_native_integration','test_fp8_z128_native_build','test_fp8_z128_stage_witness')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';sys.path[:0]=[str(repo),str(m)]
    proof=(args.proofs/(BUILD+'-verified.json')).read_bytes();assert proof==(args.proofs/(BUILD+'-remote-verified.json')).read_bytes();q=json.loads(proof);archive=args.proofs/(BUILD+'-evidence.zip');assert q['status']=='fp8_z128_native_build_cpu_replayed' and sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        w=json.loads(z.read('results/'+BUILD+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before']==w['source_after'] and w['runtime_before']==w['runtime_after'] and w['environment_before']==w['environment_after']
        raw=z.read(PREFIX+'evidence/fp8-z128-native-build-preparation.json');bp=json.loads(raw);base={**bp['base_files'],**bp['methods'],PREFIX+'evidence/fp8-z128-native-build-preparation.json':sha(raw)};bundle_raw=z.read(bp['bundle']);assert sha(bundle_raw)==bp['bundle_sha256']
    previous=(args.proofs/(PREVIOUS+'-verified.json')).read_bytes();assert previous==(args.proofs/(PREVIOUS+'-remote-verified.json')).read_bytes();pq=json.loads(previous);assert pq['status']=='gdn_chunk_stage_witness_arrays_replayed' and pq['archive_sha256']==sha((args.proofs/(PREVIOUS+'-evidence.zip')).read_bytes())
    candidates=bp['candidate_files'];assert len(candidates)==28
    for name,h in candidates.items():assert sha((repo/name).read_bytes())==h
    new={PREFIX+name:sha((m/name).read_bytes()) for name in NAMES}
    for name,h in new.items():
        if name in base:assert base[name]==h
        if name.endswith('.py'):ast.parse((repo/name.removeprefix('Garnet/')).read_bytes())
    files={**base,**new};installed={**files,**{'Garnet/'+name:h for name,h in candidates.items()}}
    def ids(suite):
        for case in suite:
            if isinstance(case,unittest.TestSuite):yield from ids(case)
            else:yield case.id()
    tests=sorted(name for module in MODULES for name in ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.'+module)));assert len(tests)==len(set(tests))==29
    import run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in candidates.items()}};ledger.source_identity=lambda:{'scope':'Explicit owned native28 Z stage preparation; unrelated untracked excluded','owned_source_sha256':pins};os.chdir(repo)
    code=ledger.run([sys.executable,'-m','unittest',*tests],args.work/'fp8-z128-stage-witness-local-cpu-ledger','correctness','29 old/new source/readback/real128/conversion/nativecontext/completearray/policyfault controls; syntheticprofiles are not hardware evidence; no local native/GPU',[repo/name.removeprefix('Garnet/') for name in pins]);assert code==0
    checkpoints={name:files[name] for name in ('models/Qwen3.8-27B-NVFP4/config.json','models/Qwen3.8-27B-NVFP4/model.safetensors.index.json')}
    p={'status':'fp8_z128_stage_witness_prepared_gpu_unrun','files':files,'new_files':new,'candidate_files':candidates,'candidate_base':bp['candidate_base'],'installed_files':installed,'runtime':bp['runtime'],'environment':bp['environment'],'checkpoints':checkpoints,'parents':{BUILD:{'status':q['status'],'exit_code':0,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']},PREVIOUS:{'status':pq['status'],'exit_code':0,'proof_sha256':sha(previous),'archive_sha256':pq['archive_sha256']}},'candidate_library_sha256':q['candidate_library_sha256'],'source_bundle':bp['bundle'],'source_bundle_sha256':bp['bundle_sha256'],'test_ids':tests,'expected_cpu_test_count':29,'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6','scope':'Exclusive closednative28typedZ stage witness:10freshpolicyfrontendprocesses (default/empty/0/1/6malformed), then8realfirst128prompt GDNlayer0/1 T32/T128ordinary/diagnostic graphs,3warmups/2repeats. Validateold/newcounters, unchangedordinaryprimary, everyinput/carriedstate/repeat/all11stage words through<=1Miword reads. Layer1commoninput fromretainedT32layer0ordinaryoutput, notfullcandidatechain. Preserveanyremainingdifference. Restore8existing/20new native targets, baseline1a1d/runtime/SDK/checkpoints/ownedSSDcache;29CPUcontrols. Full3968modelcause/805quality/HTTP/servingmetrics/2xunqualified.'}
    target=e/'fp8-z128-stage-witness-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    deploy={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};deploy[PREFIX+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=bundle_raw;package=args.work/'fp8-z128-stage-witness-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    result={'status':p['status'],'source_pins':len(files),'installed_pins':len(installed),'new_files':len(new),'candidate_files':28,'cpu_tests':29,'targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)};(args.work/'fp8-z128-stage-witness-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
