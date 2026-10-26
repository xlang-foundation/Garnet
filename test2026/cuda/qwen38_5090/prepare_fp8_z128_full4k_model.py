"""Prepare paired35source full4K model trial after exact all-eight-profile stage replay."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest();STAGE='fp8-z128-native-package-audit-attempt1';PARENT='mtp-prefill16-full4k-admission-attempt1';FAILED='bf16-prefill128-fp8-model-admission-attempt1';M='Garnet/test2026/cuda/qwen38_5090/'
NAMES=('prepare_fp8_z128_full4k_model.py','run_fp8_z128_full4k_model.py','probe_fp8_z128_full4k_model.py','test_fp8_z128_full4k_model.bat','verify_fp8_z128_full4k_model.py','fp8_z128_full4k_model.py','test_fp8_z128_full4k_model.py','evidence/fp8-z128-full4k-model-derivation.json')

def ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);args=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)];parents={};sp=None;op=None;inherited=None
    for job,status,code in ((STAGE,'fp8_z128_native_package_all8_stage_profiles_and_cold_context_replayed',0),(PARENT,'mtp_prefill16_full4k_target_state_replayed',0),(FAILED,'bf16_prefill128_fp8_full4k_model_failure_preserved',1)):
        proof=(args.proofs/(job+'-verified.json')).read_bytes();q=json.loads(proof);assert proof==(args.proofs/(job+'-remote-verified.json')).read_bytes() and q['status']==status and sha((args.proofs/(job+'-evidence.zip')).read_bytes())==q['archive_sha256']
        with zipfile.ZipFile(args.proofs/(job+'-evidence.zip')) as z:
            assert z.testzip() is None;w=json.loads(z.read('results/'+job+'/workflow.json'));assert w['exit_code']==code and w['cleanup_finished']
            if job==STAGE:
                assert w['baseline_source_restored'] and w['baseline_library_restored'] and q['array_replay']['controlled_first128_all11_stages_exact'] and q['unchanged_T32_baseline_arrays_compared']==400 and not q['post_failure_audit_invoked']
                raw=z.read(M+'evidence/fp8-z128-native-package-audit-preparation.json');sp=json.loads(raw);inherited={**sp['files'],M+'evidence/fp8-z128-native-package-audit-preparation.json':sha(raw)}
                stage_script=sha((m/'verify_fp8_z128_native_stage_success.py').read_bytes());assert stage_script==q['verifier_sha256'];inherited[M+'verify_fp8_z128_native_stage_success.py']=stage_script
                audit_names=('results/'+STAGE+'/audit-library.json','results/'+STAGE+'/build/fp8_z128_context_native.dll',sp['observer_library']);audit={name:sha(z.read(name)) for name in audit_names}
                assert audit[audit_names[1]]==q['native_package_library_sha256'] and audit[sp['observer_library']]==q['observer_library_sha256'];inherited.update(audit);library=q['candidate_library_sha256']
            elif job==PARENT:assert w['library_unchanged']
            else:
                assert w['baseline_source_restored'] and w['baseline_library_restored'];opraw=z.read(M+'evidence/bf16-prefill128-fp8-model-preparation.json');op=json.loads(opraw)
        parents[job]={'status':status,'exit_code':code,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}
    candidates=dict(sp['candidate_files']);candidate_base=dict(sp['candidate_base'])
    for name,h in op['candidate_files'].items():
        if name.startswith('xModel/'):candidates[name]=h;candidate_base[name]=op['candidate_base'][name]
    assert len(candidates)==35 and sum(h is not None for h in candidate_base.values())==12
    for name,h in candidates.items():assert sha((repo/name).read_bytes())==h
    names={name.removeprefix(M) for name in op['new_files']};names.update(NAMES)
    # These are owned historical method/provenance inputs; never enumerate/read unrelated untracked source bodies.
    assert not any('probe_native_external_prefix_batch.py' in name for name in [*names,*inherited,*candidates])
    new={M+name:sha((m/name).read_bytes()) for name in sorted(names)}
    for name,h in new.items():
        if name in inherited:assert inherited[name]==h
        if name.endswith('.py'):ast.parse((repo/name.removeprefix('Garnet/')).read_bytes())
    files={**inherited,**new}
    for name,h in candidate_base.items():
        if h is not None:assert files['Garnet/'+name]==h
    superseded=[name for name in op['test_ids'] if '.test_fp8_m128_native_source.' in name];assert len(superseded)==3
    composed=sorted(name for module in ('test_fp8_z128_native_source','test_fp8_z128_native_integration') for name in ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.'+module)));assert len(composed)==5
    model=sorted(ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_fp8_z128_full4k_model')));assert len(model)==4
    tests=sorted([name for name in op['test_ids'] if name not in superseded]+composed+model);assert len(tests)==len(set(tests))==70
    import run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in candidates.items()}};ledger.source_identity=lambda:{'scope':'Explicit native35/Python full4K model sources; unrelated untracked excluded','owned_source_sha256':pins};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*tests],args.work/'fp8-z128-full4k-model-local-cpu-ledger','correctness','70CPU retained61 model/fullstate/vector/ID/MTP controls plus5latestnativecomposition and4newpolicy/strictoracle/provenance faults; no local native/GPU',[repo/name.removeprefix('Garnet/') for name in pins])==0
    bundle=args.work/'fp8-z128-full4k-model-source.zip';assert not bundle.exists()
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for name in candidates:z.write(repo/name,name)
    p={'status':'fp8_z128_full4k_model_prepared_gpu_unrun','files':files,'inherited_files':inherited,'new_files':new,'candidate_files':candidates,'candidate_base':candidate_base,'installed_files':{**files,**{'Garnet/'+name:h for name,h in candidates.items()}},'runtime':sp['runtime'],'environment':sp['environment'],'checkpoints':sp['checkpoints'],'parents':parents,'candidate_library_sha256':library,'baseline_archive_sha256':parents[PARENT]['archive_sha256'],'source_bundle':'results/'+bundle.name,'source_bundle_sha256':sha(bundle.read_bytes()),'audit_dependencies':audit,'stage_success_verifier_sha256':stage_script,'old_model_preparation_sha256':sha(opraw),'superseded_source_test_ids':superseded,'composed_native_test_ids':composed,'new_model_test_ids':model,'test_ids':tests,'expected_cpu_test_count':70,'gpu_uuid':sp['gpu_uuid'],'scope':'Native28 closedDLL plus7unchanged target128/T4/draft16 Pythonprofile files,12existing/23new restoretransaction. Require separateoff graphconstruction botholdM128/newZ counters0; on target12848each/decode4/draft4/draft16/vision/norm0. Same six3968+128text/image/reset-repeat ordinary/MTP cases, exact128IDs/32statewitnesses/64completefinite rawvectors/callbacks versus retained1a1d T32oracle. Ordinary observations saved beforeassertion.70CPU controls retainall61oldmodel/state/vector/ID/MTP cases, replaceonly3superseded single-stage sourcechecks with5latestcomposed checks,4newchecks. Optionalnativeinit observer only afterbuildSerializedNetwork exception: create unique reportsidecar under permitted priorstudy resultroot, copyidenticalrawbytes into this trial; earlierarchives/workflows immutable. No broad805quality/HTTP/timing/throughput/2xadmission. Earlier initialization failurecause/reliability remainsunproven.'}
    target=m/'evidence/fp8-z128-full4k-model-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    deploy={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};deploy[M+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=bundle.read_bytes();package=args.work/'fp8-z128-full4k-model-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    result={'status':p['status'],'source_pins':len(files),'installed_pins':len(p['installed_files']),'new_files':len(new),'candidate_sources':35,'CPU_tests':70,'deployment_targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)};(args.work/'fp8-z128-full4k-model-package.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
