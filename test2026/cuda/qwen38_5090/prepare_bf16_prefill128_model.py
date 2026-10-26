"""Prepare the complete native/Python T128 full4K trial after closed native build replay."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
BUILD='bf16-prefill128-native-build-attempt1';PARENT='mtp-prefill16-full4k-admission-attempt1';PREFIX='Garnet/test2026/cuda/qwen38_5090/'
MODULES=('test_bf16_prefill128_runtime_contract','test_bf16_low_batch_model_contract','test_bf16_kv_model_contract','test_mtp_prefill16_study','test_mtp_prefill16_target128_study','test_mtp_prefill16_full4k','test_tiled_scores_model','test_prefill128_model')
NAMES=('probe_bf16_prefill128_model.py','test_bf16_prefill128_model.bat','run_bf16_prefill128_model.py','verify_bf16_prefill128_model.py','prepare_bf16_prefill128_model.py','prefill128_model.py','test_prefill128_model.py','test_mtp_prefill16_target128_study.py','test_bf16_prefill128_runtime_contract.py','bf16_prefill128_layout.py','evidence/prefill128-model-source-derivation.json','evidence/prefill128-native-runtime-derivation.json','evidence/prefill128-python-runtime-derivation.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';parents={};base=None;native=None;runtime=None;environment=None
    for job,status in ((BUILD,'bf16_prefill128_native_build_cpu_replayed'),(PARENT,'mtp_prefill16_full4k_target_state_replayed')):
        raw=(args.proofs/(job+'-verified.json')).read_bytes();q=json.loads(raw)
        assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes() and q['status']==status
        archive=args.proofs/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            w=json.loads(z.read('results/'+job+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished']
            if job==BUILD:
                assert w['baseline_source_restored'] and w['baseline_library_restored'] and w['source_before']==w['source_after'] and w['runtime_before']==w['runtime_after'] and w['environment_before']==w['environment_after']
                praw=z.read(PREFIX+'evidence/bf16-prefill128-native-build-preparation.json');bp=json.loads(praw)
                base={**bp['base_files'],**bp['methods'],PREFIX+'evidence/bf16-prefill128-native-build-preparation.json':sha(praw)}
                native=bp['candidate_files'];runtime=w['runtime_after'];environment=w['environment_after'];library=q['candidate_library_sha256'];native_base=bp['candidate_base']
            else:assert w['library_unchanged']
        parents[job]={'status':status,'exit_code':0,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']}
    assert len(base)==724 and len(native)==13
    candidates=dict(native);names=set(NAMES)
    for filename in ('prefill128-native-runtime-derivation.json','prefill128-python-runtime-derivation.json'):
        for row in json.loads((e/filename).read_bytes())['files']:
            raw=(repo/row['target']).read_bytes();assert sha(raw)==row['target_sha256']
            if row['target'].startswith('xModel/'):candidates[row['target']]=row['target_sha256']
            if row.get('existing_file_modified'):names.add(row['source'].removeprefix('test2026/cuda/qwen38_5090/'))
    assert len(candidates)==19
    record=json.loads((e/'prefill128-model-source-derivation.json').read_bytes())
    for row in record['files']:
        raw=(repo/row['source']).read_bytes();assert sha(raw)==row['source_sha256'];text=raw.decode()
        for step in row['replacements']:
            assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
        assert text.encode()==(repo/row['target']).read_bytes() and sha(text.encode())==row['target_sha256']
        if row['target'].startswith('xModel/'):candidates[row['target']]=row['target_sha256']
    assert len(candidates)==20
    new={PREFIX+name:sha((m/name).read_bytes()) for name in sorted(names)}
    for key,h in new.items():
        if key in base:assert base[key]==h
        if key.endswith('.py'):ast.parse((repo/key.removeprefix('Garnet/')).read_bytes())
    files={**base,**new};candidate_base={name:native_base[name] if name in native_base else base.get('Garnet/'+name) for name in candidates}
    assert sum(value is not None for value in candidate_base.values())==8
    sys.path[:0]=[str(repo),str(m)]
    def ids(suite):
        for value in suite:
            if isinstance(value,unittest.TestSuite):yield from ids(value)
            else:yield value.id()
    tests=sorted(test for module in MODULES for test in ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.'+module)))
    assert len(tests)==len(set(tests))==55
    import test2026.cuda.qwen38_5090.run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in candidates.items()}};ledger.source_identity=lambda:{'scope':'Explicit owned T128 model/routing/replay source inputs; unrelated untracked files excluded','owned_source_sha256':pins}
    os.chdir(repo);code=ledger.run([sys.executable,'-m','unittest',*tests],args.work/'prefill128-model-local-cpu-ledger','correctness','55T128 graph/factory, old/new MTP binding, strict retained4K full-state/vector/ID and partial evidence controls; no local native runtime/GPU',[repo/key.removeprefix('Garnet/') for key in pins]);assert code==0
    bundle=args.work/'bf16-prefill128-model-source.zip';assert not bundle.exists()
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for name,h in candidates.items():assert sha((repo/name).read_bytes())==h;z.write(repo/name,name)
    p={'status':'bf16_prefill128_full4k_model_prepared_gpu_unrun','files':files,'new_files':new,'candidate_files':candidates,'candidate_base':candidate_base,'installed_files':{**files,**{'Garnet/'+name:h for name,h in candidates.items()}},'runtime':runtime,'environment':environment,'parents':parents,'candidate_library_sha256':library,'baseline_archive_sha256':parents[PARENT]['archive_sha256'],'source_bundle':'results/'+bundle.name,'source_bundle_sha256':sha(bundle.read_bytes()),'test_ids':tests,'expected_cpu_test_count':55,'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6',
       'scope':'Actual B1/cap4096 complete targetT128 forward on CPU-qualified nativeDLL, retained targetT4 verification and separate exact-derivative draftT16 session with oldT32 guards unchanged. Six3968input+128output text/image/reset-repeat cases, ordinary/MTP IDs,32complete state hashes and64raw hidden/normalized vectors must match retained1a1d T32 full4K oracle exactly. Save each completed ordinary observation before any equality assertion; preserve every failure. Temporarily install20paired native/Python sources (13native/7Python), preserve8originals and12absent targets, restore all source/runtime/SDK and engine cache on exit.55CPU contracts before GPU, exclusive owner, original archived full checker replayed on both machines. No broad805quality, HTTP, timings, throughput or2x admission.'}
    target=e/'bf16-prefill128-model-preparation.json';assert not target.exists();raw=json.dumps(p,indent=2).encode();target.write_bytes(raw)
    rows=[{'local':str(m/name),'target':PREFIX+name,'sha256':new[PREFIX+name]} for name in sorted(names)]
    rows.extend(({'local':str(target),'target':PREFIX+'evidence/'+target.name,'sha256':sha(raw)},{'local':str(bundle),'target':p['source_bundle'],'sha256':p['source_bundle_sha256']}))
    (args.work/'prefill128-model-upload-plan.json').write_bytes(json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode());print(json.dumps({'status':p['status'],'baseline_pins':len(files),'installed_pins':len(p['installed_files']),'new_files':len(new),'candidate_files':20,'cpu_tests':55,'preparation_sha256':sha(raw)}));return 0

if __name__=='__main__':raise SystemExit(main())
