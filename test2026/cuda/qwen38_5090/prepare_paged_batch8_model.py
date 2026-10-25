"""Prepare whole B8 input preservation only after actual native-byte admission."""
from pathlib import Path
import argparse,ast,hashlib,json,os,sys,unittest,zipfile
COMP='paged-prompt-native-component-attempt1';BASE='bf16-batch8-model-admission-attempt1'
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--proofs',type=Path,required=True);a.add_argument('--work',type=Path,required=True);args=a.parse_args()
    args.proofs=args.proofs.resolve(strict=True);args.work=args.work.resolve(strict=True);m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parents={}
    for name,status in ((COMP,'paged_prompt_native_component_bytes_replayed'),(BASE,'bf16_batch8_internal_model_admission_replayed')):
        proof_path=args.proofs/(name+'-verified.json');q=json.loads(proof_path.read_bytes());assert q==json.loads((args.proofs/(name+'-remote-verified.json')).read_bytes()) and q['status']==status
        archive=args.proofs/(name+'-evidence.zip');assert sha(archive)==q['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            manifest=json.loads(z.read('evidence-manifest.json'));assert z.testzip() is None
            for n,h in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h
            w=json.loads(z.read('results/'+name+'/workflow.json'));assert w['exit_code']==0 and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256'] and w['runtime_dependencies']==w['runtime_dependencies_after']
            if name==COMP:
                assert q['raw_read_events']==11274 and q['logical_bytes_reconstructed_and_verified']==1703301120
                base=w['source_after_sha256'];runtime=w['runtime_dependencies_after'];environment=w['environment_after']
            else:
                reference=json.loads(z.read('results/'+name+'/model/report.json'));reference_raw={n.removeprefix('results/'+name+'/model/'):z.read(n) for n in z.namelist() if n.startswith('results/'+name+'/model/') and n.endswith('.f32')}
                retained_b8={}
                for filename in ('bf16_batch8_admission.py','verify_bf16_batch8_admission.py','test_bf16_batch8_admission.py','test_bf16_batch8_replay.py'):
                    path='Garnet/test2026/cuda/qwen38_5090/'+filename;digest=w['source_after_sha256'][path]
                    assert hashlib.sha256(z.read(path)).hexdigest()==digest==sha(m/filename)
                    retained_b8[path]=digest
        parents[name]={'status':status,'proof_sha256':sha(proof_path),'archive_sha256':q['archive_sha256']}
    names=('paged_batch8_model.py','test_paged_batch8_model.py','probe_paged_batch8_model.py','test_paged_batch8_model.bat','run_paged_batch8_model.py','verify_paged_batch8_model.py','prepare_paged_batch8_model.py','evidence/paged-batch8-model-probe-derivation.json')
    new={'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in names};assert not set(new)&set(base)
    for n in names:
        if n.endswith('.py'):ast.parse((m/n).read_text())
    candidate=json.loads((e/'bf16-scores-tiled-native-api-build-preparation.json').read_bytes())['candidate_files']
    inherited={**base,**retained_b8}
    for n,h in inherited.items():
        if n.startswith('Garnet/'):
            name=n.removeprefix('Garnet/');assert sha(repo/name)==(candidate[name] if name in candidate else h),('Baseline/candidate source drift',name)
    sys.path.insert(0,str(repo));from test2026.cuda.qwen38_5090.run_logged import run
    from test2026.cuda.qwen38_5090.paged_batch8_model import replay_preserved
    def ids(suite):
        for t in suite:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    modules=['test2026.cuda.qwen38_5090.'+n for n in ('test_bf16_batch8_admission','test_bf16_batch8_replay','test_paged_batch8_model','test_paged_prompt_rows','test_paged_prompt_component')]
    tests=sorted(t for name in modules for t in ids(unittest.defaultTestLoader.loadTestsFromName(name)));assert len(tests)==26
    assert all('Garnet/'+name.replace('.','/')+'.py' in {**inherited,**new} for name in modules)
    os.chdir(repo);code=run([sys.executable,'-m','unittest',*tests],args.work/'paged-batch8-model-cpu-ledger','correctness','26source-pinned old/new B8 state and raw-vector/shared-drift/paged input controls',[m/n for n in names]);assert code==0
    import copy
    control=copy.deepcopy(reference)
    for c in control['cases']:
        lengths=[len(v['prompt_ids']) for v in c['inputs']];count=max(lengths);c['prompt_cache']={'physical_batch':8,'tokens':count,'lengths':lengths,'page_size':128,
            'pack_calls':(count+127)//128,'cached_pages':[0],'peak_cached_packed_logical_bytes':min(2*128,count)*8*20480}
    replay=replay_preserved(control,reference_raw.__getitem__,reference,reference_raw.__getitem__)
    (args.work/'paged-batch8-retained-cpu-replay.json').write_bytes(json.dumps({'status':'actual_retained_b8_states_replayed_cpu_only','replay':replay,'gpu_executed':False},indent=2).encode())
    p={'status':'paged_batch8_model_prepared_gpu_unrun','files':{**inherited,**new},'new_files':new,'retained_b8_source_files':retained_b8,'runtime':runtime,'environment':environment,'parents':parents,
       'baseline_archive_sha256':parents[BASE]['archive_sha256'],'test_ids':tests,'gpu_uuid':'GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6',
       'scope':'Only replace full-prompt embeddings/copied rows/stacked tensors with admitted128two-page input source in existing baseline1a1d B8 BF16-cache/T32prefill/T4decode model. Same nine short/full3968+128text/image/original/rotated/reset cases, original decoder/vision/generation arithmetic unchanged. Require every input/token/complete canonical state and raw finite hidden/normalized vector match retained actual B8 baseline;29states/58vectors. Observe actual construction/correctness adapter memory with raw samples/gaps, not serving peak or causal savings. Baseline source/runtime/SDK unchanged. Existing B8 external B1/B4 math, MTP,805quality/HTTP/throughput/2x remain unqualified.'}
    target=e/'paged-batch8-model-preparation.json';assert not target.exists();target.write_bytes(json.dumps(p,indent=2).encode())
    rows=[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in names]
    rows.append({'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)})
    (args.work/'paged-batch8-model-upload-plan.json').write_bytes(json.dumps({'uploads':rows},indent=2).encode());print(json.dumps({'status':p['status'],'source_pins':len(p['files']),'new_targets':len(rows),'cpu_tests':26,'preparation_sha256':sha(target)}));return 0
if __name__=='__main__':raise SystemExit(main())
