"""Pin preserved strict failure and existing B8 model before same-B4 cache diagnosis."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()
METHODS=('b4_cache_mode_characterization.py','test_b4_cache_mode_characterization.py','probe_b4_cache_mode_characterization.py','test_b4_cache_mode_characterization.bat',
 'run_b4_cache_mode_characterization.py','verify_b4_cache_mode_characterization.py','prepare_b4_cache_mode_characterization.py','evidence/b4-cache-mode-characterization-derivation.json')

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    m=Path(__file__).resolve().parent;repo=m.parents[2];parents={};base=None;old_praw=None
    for job,status,exit_code in (('paged-batch8-model-preservation-attempt1','paged_batch8_model_preservation_replayed',0),('bf16-batch8-external-reference-attempt1','bf16_batch8_external_reference_failure_preserved',1)):
        raw=(args.proofs/(job+'-verified.json')).read_bytes();q=json.loads(raw);assert raw==(args.proofs/(job+'-remote-verified.json')).read_bytes() and q['status']==status
        archive=args.proofs/(job+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            w=json.loads(z.read('results/'+job+'/workflow.json'));assert type(w['exit_code']) is int and w['exit_code']==exit_code and w['cleanup_finished'] is True
            assert w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after'] and w['environment_before']==w['environment_after']
            if exit_code==1:
                base=w;old_praw=z.read('Garnet/test2026/cuda/qwen38_5090/evidence/bf16-batch8-external-reference-preparation.json')
        parents[job]={'status':status,'exit_code':exit_code,'proof_sha256':sha(raw),'archive_sha256':q['archive_sha256']}
    files=dict(base['source_after_sha256']);old=json.loads(old_praw);assert len(old['files'])==697
    files['Garnet/test2026/cuda/qwen38_5090/evidence/bf16-batch8-external-reference-preparation.json']=sha(old_praw)
    new={'Garnet/test2026/cuda/qwen38_5090/'+name:sha((m/name).read_bytes()) for name in METHODS}
    assert not set(files)&set(new);files.update(new)
    for row in json.loads((m/'evidence/b4-cache-mode-characterization-derivation.json').read_bytes())['files']:
        raw=(m/row['source']).read_bytes();assert sha(raw)==row['source_sha256'];text=raw.decode()
        for step in row['replacements']:
            assert text.count(step['old'])==step['count'];text=text.replace(step['old'],step['new'])
        assert text.encode()==(m/row['target']).read_bytes() and sha(text.encode())==row['target_sha256']
    for name in METHODS:
        if name.endswith('.py'):ast.parse((m/name).read_bytes())
    sys.path[:0]=[str(repo),str(m)]
    module='test2026.cuda.qwen38_5090.test_b4_cache_mode_characterization';suite=unittest.defaultTestLoader.loadTestsFromName(module)
    def ids(suite):
        for test in suite:
            if isinstance(test,unittest.TestSuite):yield from ids(test)
            else:yield test.id()
    tests=sorted(old['test_ids']+list(ids(suite)));assert len(set(tests))==len(tests)==50
    import test2026.cuda.qwen38_5090.run_logged as ledger
    ledger.source_identity=lambda:{'scope':'Explicit owned new local methods; baseline source pins are the preserved remote archive, excluding unrelated untracked files','owned_source_sha256':new}
    os.chdir(repo);code=ledger.run([sys.executable,'-m','unittest',*tests],args.work/'b4-cache-mode-characterization-cpu-ledger','correctness','Existing45controls plus5standard-library exact-state/vector/ID diagnostic faults; no local native runtime/GPU',[m/name for name in METHODS]);assert code==0
    p={'status':'b4_cache_mode_characterization_prepared_gpu_unrun','files':files,'new_files':new,'runtime':base['runtime_dependencies_after'],'environment':base['environment_after'],'parents':parents,
       'test_ids':tests,'expected_cpu_test_count':50,'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6',
       'scope':'Repeat only the actual first four-row short8-output case, sequentially with existing B4 full-FP32-cache/BF16-values and B4 packed-BF16 cache on baseline1a1d. Exact same checkpoint/source/runtime/request bytes/IDs and retained complete expanded logical-state hashes/raw hidden-normalized vectors. Compare both to actual closed B8 and repeat-full to preserved failed B4; differences are observations, no exact gate relaxed or cause/harmlessness/semantic/mathematical/quality/performance equivalence admitted. Independent same-source raw replay and exclusive cache cleanup required. T128 local forward candidate is not installed or involved. No MTP,805screen,HTTP or serving timing/VRAM claim.'}
    target=m/'evidence/b4-cache-mode-characterization-preparation.json';raw=json.dumps(p,indent=2).encode();assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    rows=[{'local':str(m/name),'target':'Garnet/test2026/cuda/qwen38_5090/'+name,'sha256':sha((m/name).read_bytes())} for name in METHODS]
    rows.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(raw)})
    (args.work/'b4-cache-mode-characterization-upload-plan.json').write_bytes(json.dumps({'uploads':rows,'scope':p['scope']},indent=2).encode())
    print(json.dumps({'status':p['status'],'source_pins':len(files),'cpu_tests':50,'preparation_sha256':sha(raw)}));return 0

if __name__=='__main__':raise SystemExit(main())
