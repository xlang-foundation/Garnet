"""Prepare the next I/O trial only from an actually closed two-machine model proof."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile

PARENT='bf16-scores-tiled-api-model-admission-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    args.proofs=args.proofs.resolve(strict=True);args.work=args.work.resolve(strict=True)
    m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    needed=[args.proofs/(PARENT+s) for s in ('-verified.json','-remote-verified.json','-evidence.zip')]
    if not all(f.is_file() for f in needed):
        print(json.dumps({'status':'paged_prompt_preparation_waiting_for_actual_parent_closure','missing':[str(f) for f in needed if not f.is_file()],'gpu_executed':False}));return 2
    local,remote,archive=needed;q=json.loads(local.read_bytes());assert q==json.loads(remote.read_bytes()) and q['status'] in ('native_tiled_scores_model_replayed','native_tiled_scores_model_failure_preserved') and sha(archive)==q['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
        for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        w=json.loads(z.read('results/'+PARENT+'/workflow.json'))
    assert type(w['exit_code']) is int and w['cleanup_finished'] and w['baseline_library_restored'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
    base=w['source_after_sha256'];runtime=w['runtime_dependencies_after'];assert runtime['garnet.dll']=='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
    candidate=json.loads((e/'bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    for n,h in base.items():
        if n.startswith('Garnet/'):
            name=n.removeprefix('Garnet/');actual=sha(repo/name)
            assert actual==(candidate[name] if name in candidate else h),('Local baseline/candidate source drift',name)
    names=('paged_prompt_rows.py','test_paged_prompt_rows.py','paged_prompt_component.py','test_paged_prompt_component.py','probe_paged_prompt_component.py',
           'probe_paged_prompt_embedded_cpu.py','run_paged_prompt_component.py','test_paged_prompt_component.bat','verify_paged_prompt_component.py','prepare_paged_prompt_component.py')
    new={'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in names};assert not set(new)&set(base)
    for n in names:
        if n.endswith('.py'):ast.parse((m/n).read_text())
    fixture_archive=args.proofs/'host-embedding-rows-attempt1-evidence.zip';assert sha(fixture_archive)=='7a7a7effcd1af201f103b5df5f7a74ef4b8aca67eda5c2c219eea86e57d75801'
    fixture_files={}
    with zipfile.ZipFile(fixture_archive) as z:
        prefix='results/host-embedding-rows-attempt1/fixture/';f=json.loads(z.read(prefix+'fixture.json'));assert f['shape']==[248320,5120]
        filenames=['fixture.json']+[entry[k] for entry in f['rows'].values() for k in ('bf16','f32')]
        assert len(filenames)==len(set(filenames))==31
        for n in filenames:
            assert '/' not in n and '\\' not in n
            fixture_files[prefix+n]=hashlib.sha256(z.read(prefix+n)).hexdigest()
    sys.path.insert(0,str(repo));from test2026.cuda.qwen38_5090.run_logged import run
    def ids(suite):
        for t in suite:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    modules=('test2026.cuda.qwen38_5090.test_paged_prompt_rows','test2026.cuda.qwen38_5090.test_paged_prompt_component')
    tests=sorted(t for name in modules for t in ids(unittest.defaultTestLoader.loadTestsFromName(name)));assert len(tests)==12
    os.chdir(repo);code=run([sys.executable,'-m','unittest',*tests],args.work/'paged-prompt-native-preparation-ledger','correctness',
                            'Twelve source-pinned bounded input and raw-byte replay fault controls before GPU preparation',[m/n for n in names]+needed);assert code==0
    p={'status':'paged_prompt_native_component_prepared_gpu_unrun','files':{**base,**new},'new_files':new,'runtime':runtime,'environment':w['environment_after'],
       'retained_fixture_files':fixture_files,'closed_native_proof_sha256':sha(local),'closed_native_archive_sha256':q['archive_sha256'],'test_ids':tests,
       'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6',
       'scope':'Eight native-I/O cases: B1/2/4/8, pages32/64/128, unequal lengths/padding, exact64x5120synthetic visuals crossing boundaries/new visual epoch/permutation, B8/3968 full input. Actual twelve CPU controls in both CPython and embedded runtime before GPU. Reprepare independent checkpoint BF16 row fixture and bind original retained31files. Retain raw observed row bytes once with ordered events, reconstruct every byte of11274reads independently; test held views after eviction/close. Baseline1a1d/source/RTE/SDK unchanged, no decoder constructed. Resource memory is component construction/correctness only. No B8model/MTP/quality/HTTP/serving-memory-saving/speed or2x admission.'}
    target=e/'paged-prompt-native-component-preparation.json';assert not target.exists();target.write_bytes(json.dumps(p,indent=2).encode())
    rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in names]
    rows.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)})
    (args.work/'paged-prompt-native-component-upload-plan.json').write_bytes(json.dumps({'uploads':rows},indent=2).encode())
    print(json.dumps({'status':p['status'],'source_pins':len(p['files']),'new_targets':len(rows),'preparation_sha256':sha(target),'gpu_executed':False}));return 0
if __name__=='__main__':raise SystemExit(main())
