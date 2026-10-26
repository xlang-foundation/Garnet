"""CPU-only preparation for the closed native diagnostic candidate; no local GPU."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
BUILD='gdn-chunk-stage-native-build-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/';sha=lambda raw:hashlib.sha256(raw).hexdigest()
NAMES=('prepare_gdn_chunk_stage_witness.py','run_gdn_chunk_stage_witness.py','probe_gdn_chunk_stage_witness.py','verify_gdn_chunk_stage_witness.py','test_gdn_chunk_stage_witness.bat','gdn_chunk_stage_layout.py','gdn_chunk_stage_replay.py','test_gdn_chunk_stage_witness.py','evidence/gdn-chunk-stage-witness-owner-derivation.json','evidence/native-host-stage-cases.json')
MODULES=('test_gdn_chunk_stage_source','test_gdn_chunk_stage_native_build','test_prefill128_fp8_projection','test_prefill128_fp8_workspace8','test_gdn_chunk_stage_witness')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';sys.path[:0]=[str(repo),str(m)]
    proof=(args.proofs/(BUILD+'-verified.json')).read_bytes();assert proof==(args.proofs/(BUILD+'-remote-verified.json')).read_bytes();q=json.loads(proof);archive=args.proofs/(BUILD+'-evidence.zip');assert q['status']=='gdn_chunk_stage_native_build_cpu_replayed' and sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        w=json.loads(z.read('results/'+BUILD+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before']==w['source_after'] and w['runtime_before']==w['runtime_after'] and w['environment_before']==w['environment_after']
        raw=z.read(PREFIX+'evidence/gdn-chunk-stage-native-build-preparation.json');bp=json.loads(raw);base={**bp['base_files'],**bp['methods'],PREFIX+'evidence/gdn-chunk-stage-native-build-preparation.json':sha(raw)}
    candidates=bp['candidate_files'];assert len(candidates)==23
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
    tests=sorted(name for module in MODULES for name in ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.'+module)));assert len(tests)==len(set(tests))==18
    import test2026.cuda.qwen38_5090.run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in candidates.items()}};ledger.source_identity=lambda:{'scope':'Explicit owned GDN primary witness preparation inputs; unrelated untracked files excluded','owned_source_sha256':pins};os.chdir(repo)
    code=ledger.run([sys.executable,'-m','unittest',*tests],args.work/'gdn-chunk-stage-witness-local-cpu-ledger','correctness','18 CPU composed source/profile/oldquantization and full-array piece/primary/hash/finite/graph metadata controls; synthetic8profile data is not hardware/model evidence. No local native runtime/GPU',[repo/name.removeprefix('Garnet/') for name in pins]);assert code==0
    bundle=args.work/'gdn-chunk-stage-witness-source.zip';assert not bundle.exists()
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for name in candidates:z.write(repo/name,name)
    fixture=json.loads((e/'prefill128-fp8-workspace8-attempt1-real-fixture-manifest.json').read_bytes());checkpoints={'models/Qwen3.8-27B-NVFP4/config.json':fixture['config_sha256'],'models/Qwen3.8-27B-NVFP4/model.safetensors.index.json':fixture['index_sha256']};assert files['models/Qwen3.8-27B-NVFP4/config.json']==fixture['config_sha256']
    p={'status':'gdn_chunk_stage_witness_prepared_gpu_unrun','files':files,'new_files':new,'candidate_files':candidates,'candidate_base':bp['candidate_base'],'installed_files':installed,'runtime':bp['runtime'],'environment':bp['environment'],'checkpoints':checkpoints,'parents':{BUILD:{'status':q['status'],'exit_code':0,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}},'candidate_library_sha256':q['candidate_library_sha256'],'source_bundle':'results/'+bundle.name,'source_bundle_sha256':sha(bundle.read_bytes()),'test_ids':tests,'expected_cpu_test_count':18,'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6','scope':'Exclusive real first128frozen textprompt GDN layer0/1 witness with shared calibrated mixed projections and23native source profile. First actualfrontend10reject controls, then8ordinary/diagnostic B1T32/T128profiles,3warmups/2repeats with stablebuffers. Require full primary hidden/history/state parity before stage attribution, everyinput/carriedstate/repeat word andall11stage arrays with <=1Miword pieces. Layer1usescommonretainedT32layer0ordinaryinput, not fullcandidatechain. Restore8existing/15new source targets, baseline1a1d/runtime/SDK/cache. Pin config/index and nativecandidate;18CPU controls. Full3968model cause,805quality,HTTP/servingmemory/timing/2xremain unqualified.'}
    target=e/'gdn-chunk-stage-witness-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    deploy={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};deploy[PREFIX+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=bundle.read_bytes();package=args.work/'gdn-chunk-stage-witness-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    result={'status':p['status'],'source_pins':len(files),'installed_pins':len(installed),'new_files':len(new),'candidate_files':23,'cpu_tests':18,'targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)};(args.work/'gdn-chunk-stage-witness-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
