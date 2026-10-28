"""Prepare paired29source model trial from closed full native build, CPU only."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
BUILD='bf16-prefill128-fp8-native-build-attempt1';PARENT='mtp-prefill16-full4k-admission-attempt1';FAILED='bf16-prefill128-model-admission-attempt2';PREFIX='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';parents={};bp=None;base=None
    for job,status,exit_code in ((BUILD,'bf16_prefill128_fp8_native_build_cpu_replayed',0),(PARENT,'mtp_prefill16_full4k_target_state_replayed',0),(FAILED,'bf16_prefill128_full4k_model_failure_preserved',1)):
        raw=(args.proofs/(job+'-verified.json')).read_bytes();assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes();q=json.loads(raw);assert q['status']==status
        archive=args.proofs/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            w=json.loads(z.read('results/'+job+'/workflow.json'));assert w['exit_code']==exit_code and w['cleanup_finished']
            if job==BUILD:
                assert w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before']==w['source_after'] and w['runtime_before']==w['runtime_after'] and w['environment_before']==w['environment_after']
                praw=z.read(PREFIX+'evidence/bf16-prefill128-fp8-native-build-preparation.json');bp=json.loads(praw);base={**bp['base_files'],**bp['methods'],PREFIX+'evidence/bf16-prefill128-fp8-native-build-preparation.json':sha(praw)};library=q['candidate_library_sha256']
            elif job==PARENT:assert w['library_unchanged']
            else:assert w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
        parents[job]={'status':status,'exit_code':exit_code,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']}
    old=json.loads((e/'bf16-prefill128-model-bound-preparation.json').read_bytes());candidates=dict(bp['candidate_files']);candidate_base=dict(bp['candidate_base'])
    assert len(candidates)==22
    for name,h in old['candidate_files'].items():
        if name.startswith('xModel/'):
            candidates[name]=h;candidate_base[name]=old['candidate_base'][name]
    assert len(candidates)==len(candidate_base)==29 and sum(value is not None for value in candidate_base.values())==12
    for name,h in candidates.items():assert sha((repo/name).read_bytes())==h
    names={name.removeprefix(PREFIX) for name in old['new_files']};assert len(names)==33
    names.update(('prepare_bf16_prefill128_fp8_model.py','run_bf16_prefill128_fp8_model.py','probe_bf16_prefill128_fp8_model.py','test_bf16_prefill128_fp8_model.bat','verify_bf16_prefill128_fp8_model.py','prefill128_fp8_model.py','test_prefill128_fp8_model.py','evidence/prefill128-fp8-model-source-derivation.json','test_fp8_m128_native_source.py','evidence/fp8-m128-native-source-before.json','evidence/fp8-m128-native-source-derivation.json'))
    native=json.loads((e/'fp8-m128-native-source-derivation.json').read_bytes())
    for row in native['modified_files']:names.add(row['original_snapshot'].removeprefix('test2026/cuda/qwen38_5090/'))
    from test2026.cuda.qwen38_5090.prepare_bf16_prefill128_model_bound import MODULES,assert_provenance_closure
    assert_provenance_closure(m,names)
    for name in names:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    new={PREFIX+name:sha((m/name).read_bytes()) for name in sorted(names)}
    for key,h in new.items():
        if key in base:assert base[key]==h
    files={**base,**new}
    for name,h in candidate_base.items():
        if h is not None:assert files['Garnet/'+name]==h
    sys.path[:0]=[str(repo),str(m)]
    def ids(suite):
        for test in suite:
            if isinstance(test,unittest.TestSuite):yield from ids(test)
            else:yield test.id()
    tests=sorted(test for module in MODULES+('test_fp8_m128_native_source','test_prefill128_fp8_model') for test in ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.'+module)) if test!='test2026.cuda.qwen38_5090.test_bf16_prefill128_runtime_contract.Controls.test_recorded_native_and_python_source_transformations')
    assert len(tests)==len(set(tests))==64
    import test2026.cuda.qwen38_5090.run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in candidates.items()}};ledger.source_identity=lambda:{'scope':'Explicit owned paired M128 FP8 model sources; unrelated untracked files excluded','owned_source_sha256':pins}
    os.chdir(repo);code=ledger.run([sys.executable,'-m','unittest',*tests],args.work/'prefill128-fp8-model-local-cpu-ledger','correctness','57original model/routing/MTP/fullstate/ID and dependency controls,3exact composed native source checks,4new actual-selection/old-oracle/source-derivation controls. Original single-stage provenance test untouched. No local native runtime/GPU',[repo/key.removeprefix('Garnet/') for key in pins]);assert code==0
    bundle=args.work/'bf16-prefill128-fp8-model-source.zip';assert not bundle.exists()
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for name in candidates:z.write(repo/name,name)
    p={'status':'bf16_prefill128_fp8_full4k_model_prepared_gpu_unrun','files':files,'new_files':new,'candidate_files':candidates,'candidate_base':candidate_base,'installed_files':{**files,**{'Garnet/'+name:h for name,h in candidates.items()}},'runtime':bp['runtime'],'environment':bp['environment'],'parents':parents,'candidate_library_sha256':library,'baseline_archive_sha256':parents[PARENT]['archive_sha256'],'source_bundle':'results/'+bundle.name,'source_bundle_sha256':sha(bundle.read_bytes()),'test_ids':tests,'expected_cpu_test_count':64,'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6','scope':'Actual paired29source B1/cap4096 T128target/T4verification/T16draft candidate on closed native build.12existing target bytes restored/17owned new targets removed. Separate switch0 construction asserts target counters0/0 then close; switch1 asserts targetT128counter48/T4counter0 and draft4/draft16/vision/anynorm counters0. Six3968+128text/image/reset-repeat ordinary/MTP cases preserve original exact128IDs/32states/64raw vectors versus1a1d T32 oracle. Save completed ordinary observations and selection metadata before assertions; preserve every failure.64CPU controls, exclusive source-frozen GPU owner, unchanged runtime/SDK/fixtures/cache restoration; full archive replay both machines. No broad805quality, HTTP, TTFT, servingVRAM, throughput or2x admission.'}
    target=e/'bf16-prefill128-fp8-model-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    deploy={key:(repo/key.removeprefix('Garnet/')).read_bytes() for key in new};deploy[PREFIX+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=bundle.read_bytes()
    package=args.work/'bf16-prefill128-fp8-model-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    result={'status':p['status'],'files':len(files),'installed_files':len(p['installed_files']),'new_files':len(new),'candidate_files':29,'cpu_tests':64,'deployment_targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)}
    (args.work/'prefill128-fp8-model-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0

if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]));raise SystemExit(main())
