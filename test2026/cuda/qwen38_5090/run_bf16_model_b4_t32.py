"""Run the explicit BF16 B4/T32prefill/T4decode model gate after default-off replay."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');repo=root/'Garnet';results=root/'results';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods));from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
name='bf16-model-b4-t32-attempt1';out=results/name;assert not out.exists()
buildname='bf16-model-integration-build-attempt1';proofpath=results/(buildname+'-verified.json');proof=json.loads(proofpath.read_bytes())
assert proof['status']=='bf16_model_native_build_replayed' and proof['cpu_tests']==33
assert sha((results/(buildname+'-evidence.zip')).read_bytes())==proof['archive_sha256']
assert sha((methods/'verify_bf16_model_build.py').read_bytes())==proof['replay_verifier_sha256']
bpath=results/buildname/'manifest.json';b=json.loads(bpath.read_bytes());sources=b['candidate_source_sha256']
for n,h in sources.items():assert sha((repo/n).read_bytes())==h,('Source drift',n)
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==proof['library_sha256']==b['library_sha256']
parent='bf16-model-default-off-attempt1';parentpath=results/(parent+'-verified.json');parentproof=json.loads(parentpath.read_bytes())
assert parentproof['status']=='bf16_model_default_off_replayed' and parentproof['raw_last_hidden_all_exact']
assert sha((results/(parent+'-evidence.zip')).read_bytes())==parentproof['archive_sha256']
with zipfile.ZipFile(results/'native-kv-delta-default-off-attempt1-evidence.zip') as z:
    prior=json.loads(z.read('results/native-kv-delta-default-off-attempt1/workflow.json'))
sources={**prior['source_sha256'],**sources}
for n,h in sources.items():assert sha((repo/n).read_bytes())==h,('Dependency drift',n)
probe=methods/'probe_bf16_model_b4_t32.py';launcher=methods/'test_bf16_model_b4_t32.bat'
prep_path=methods/'evidence/bf16-model-b4-t32-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['methods'].items():assert sha((methods/n).read_bytes())==h,('Probe method drift',n)
for n,h in prep['pinned_inputs'].items():assert sha((root/n).read_bytes())==h,('Input drift',n)
fixture=results/'bf16-kv-metadata-fixture.json';assert sha(fixture.read_bytes())==prep['fixture_sha256']
config=root/'models/Qwen3.8-27B-NVFP4/config.json';assert sha(config.read_bytes())==prep['model_config_sha256']
recipe=json.loads((methods/'evidence/bf16-model-b4-t32-probe-derivation.json').read_bytes())
original=methods/recipe['source'];assert sha(original.read_bytes())==recipe['source_sha256']
derived=original.read_text(encoding='utf8')
for before,after in recipe['changes']:assert derived.count(before)==1;derived=derived.replace(before,after)
assert derived.encode()==probe.read_bytes() and sha(probe.read_bytes())==recipe['result_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time();os.chdir(repo)
w={'status':'running','library_sha256':digest,'build_proof_sha256':sha(proofpath.read_bytes()),'build_archive_sha256':proof['archive_sha256'],
   'parent_archive_sha256':parentproof['archive_sha256'],'source_sha256':sources,'probe_sha256':sha(probe.read_bytes()),
   'bf16_kv_enabled':True,'preparation_sha256':sha(prep_path.read_bytes()),'fixture_sha256':prep['fixture_sha256'],'model_config_sha256':prep['model_config_sha256'],'cleanup_finished':False,'scope':prep['scope']}
def save():(out/'workflow.json').write_text(json.dumps(w,indent=2),encoding='utf8')
save();code=1
try:
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],out/'ledger','correctness','BF16 B4/T32prefill/T4decode helper and full-model integration',
        [library,probe,launcher,proofpath,parentpath,receipt,Path(__file__),prep_path,fixture,config]+[methods/n for n in prep['methods']]+[repo/n for n in sources])
    if code:raise RuntimeError('BF16 B4/T32prefill/T4decode model gate failed')
    w['status']='bf16_b4_t32_finished_needs_replay'
except BaseException as error:
    w.update(status='failed',error=repr(error));code=code or 1;raise
finally:
    w.update(exit_code=code,library_unchanged=sha(library.read_bytes())==digest,source_after_sha256={n:sha((repo/n).read_bytes()) for n in sources});save()
    idle();retired=[];resolved=cache.resolve(strict=True)
    assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root)
    for target in cache.iterdir():
        if target.name in before or not target.name.startswith('garnet-cuda-graph-'):continue
        st=target.lstat();assert not(getattr(st,'st_file_attributes',0)&0x400) and target.resolve(strict=True).parent==resolved and st.st_ctime>=started-2
        entries=list(target.iterdir());assert all(p.name in ('model.engine','model.engine.tmp') for p in entries)
        for p in entries:
            ps=p.lstat();assert p.is_file() and not(getattr(ps,'st_file_attributes',0)&0x400) and p.resolve(strict=True).parent==target.resolve(strict=True)
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8<<20),b''):h.update(block)
            retired.append({'path':str(p),'bytes':ps.st_size,'sha256':h.hexdigest()})
        for p in entries:p.unlink()
        target.rmdir()
    w['retired_private_engine_files']=retired;w['cleanup_finished']=True;save()
print(json.dumps(w,indent=2));raise SystemExit(code)
