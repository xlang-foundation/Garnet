"""Prepare read-only ordinary controls linked to the closed full-model failure."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest();M='Garnet/test2026/cuda/qwen38_5090/'
PARENT='fp8-z128-full4k-model-admission-attempt1'
NAMES=('fa3_prefix_cache_witness.py','test_fa3_prefix_cache_witness.py','probe_fa3_prefix_cache_witness.py','test_fa3_prefix_cache_witness.bat','prepare_fa3_prefix_cache_witness.py','run_fa3_prefix_cache_witness.py','verify_fa3_prefix_cache_witness.py','evidence/fa3-prefix-cache-witness-derivation.json')
def test_ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from test_ids(item)
        else:yield item.id()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--work',type=Path,required=True);a=parser.parse_args();m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path[:0]=[str(repo),str(m)]
    proof=(a.proofs/(PARENT+'-verified.json')).read_bytes();analysis=(a.proofs/(PARENT+'-analysis.json')).read_bytes();q=json.loads(proof);failure=json.loads(analysis)
    assert proof==(a.proofs/(PARENT+'-remote-verified.json')).read_bytes() and analysis==(a.proofs/(PARENT+'-remote-analysis.json')).read_bytes()
    assert q['status']=='fp8_z128_full4k_model_failure_preserved' and failure['source_runtime_environment_restored'] and failure['first_different_output_zero_based']==57 and failure['original_failure_proof_sha256']==sha(proof)
    archive=a.proofs/(PARENT+'-evidence.zip');assert sha(archive.read_bytes())==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'))
        for name,h in manifest['files'].items():assert sha(z.read(name))==h
        praw=z.read(M+'evidence/fp8-z128-full4k-model-preparation.json');p=json.loads(praw);w=json.loads(z.read('results/'+PARENT+'/workflow.json'))
        assert w['exit_code']==1 and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored']
    parents={**p['parents'],PARENT:{'status':q['status'],'exit_code':1,'proof_sha256':sha(proof),'archive_sha256':q['archive_sha256']}}
    inherited={**p['files'],M+'evidence/fp8-z128-full4k-model-preparation.json':sha(praw)}
    new={**p['new_files'],**{M+name:sha((m/name).read_bytes()) for name in NAMES}}
    for name in ('analyze_fp8_z128_full4k_model_failure.py','evidence/fp8-z128-full4k-model-analysis-derivation.json'):
        new[M+name]=sha((m/name).read_bytes())
    files={**inherited,**new};assert not any('probe_native_external_prefix_batch.py' in name for name in files)
    for name,h in new.items():
        raw=(repo/name.removeprefix('Garnet/')).read_bytes();assert sha(raw)==h
        if name.endswith('.py'):ast.parse(raw)
    tests=p['test_ids']+sorted(test_ids(unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.test_fa3_prefix_cache_witness')));assert len(tests)==len(set(tests))==74
    import run_logged as ledger
    pins={**new,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}};ledger.source_identity=lambda:{'scope':'Explicit owned cache-observer and native35 source; unrelated untracked excluded','owned_source_sha256':pins};os.chdir(repo)
    assert ledger.run([sys.executable,'-m','unittest',*tests],a.work/'fa3-prefix-cache-witness-local-cpu-ledger','correctness','Retain70 model/state/vector/ID/MTP controls plus4 cache geometry/read/finite/index/oracle fault tests; CPU only',[repo/name.removeprefix('Garnet/') for name in pins])==0
    bundle=a.work/'fp8-z128-full4k-model-source.zip';assert sha(bundle.read_bytes())==p['source_bundle_sha256']
    result={**p,'status':'fa3_prefix_cache_witness_prepared_gpu_unrun','files':files,'inherited_files':inherited,'new_files':new,'installed_files':{**files,**{'Garnet/'+name:h for name,h in p['candidate_files'].items()}},'parents':parents,'test_ids':tests,'expected_cpu_test_count':74,'parent_preparation_sha256':sha(praw),'failure_analysis_sha256':sha(analysis),'scope':'Same native35/DLL, no math changes. Two exclusive ordinary target-only processes T32/T128, six complete layer3 key/value cache prefix reads at128/256/512/1024/2048/3968 using <=1Miword pieces. Require exact empty/prefill/final complete state and5120hidden/normalized vectors and128IDs against closed respective controls before attribution. Preserve failed captures and restore12old/23new source/runtime/SDK/checkpoint/cache. No MTP/image/quality/timing/performance/2x qualification.'}
    target=m/'evidence/fa3-prefix-cache-witness-preparation.json';assert not target.exists();raw=json.dumps(result,indent=2).encode();target.write_bytes(raw)
    deploy={name:(repo/name.removeprefix('Garnet/')).read_bytes() for name in new};deploy[M+'evidence/'+target.name]=raw;deploy[p['source_bundle']]=bundle.read_bytes();deploy[M+'evidence/fp8-z128-full4k-model-preparation.json']=praw
    package=a.work/'fa3-prefix-cache-witness-deployment.zip';assert not package.exists()
    with zipfile.ZipFile(package,'x',zipfile.ZIP_DEFLATED) as z:
        for name,value in deploy.items():z.writestr(name,value)
        z.writestr('install-manifest.json',json.dumps({'files':{name:sha(value) for name,value in deploy.items()}},indent=2))
    summary={'status':result['status'],'source_pins':len(files),'installed_pins':len(result['installed_files']),'new_files':len(new),'candidate_sources':35,'CPU_tests':74,'deployment_targets':len(deploy),'deployment_sha256':sha(package.read_bytes()),'bytes':package.stat().st_size,'preparation_sha256':sha(raw)}
    (a.work/'fa3-prefix-cache-witness-package.json').write_bytes(json.dumps(summary,indent=2).encode());print(json.dumps(summary));return 0
if __name__=='__main__':raise SystemExit(main())
