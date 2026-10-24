"""Guarded five-module compact-model installation and full correctness run.

No native rebuild or serving launch. Preserve original copied source and stage
bytes; leave failed candidate evidence intact for inspection before any retry.
"""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='native-kv-delta-default-off-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='native_kv_delta_default_off_replayed' and proof['raw_last_hidden_arrays_compared']==8 and proof['raw_last_hidden_all_exact'] and len(proof['cases'])==4
archive=results/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
closed=[json.loads(l) for l in (results/parent/'ledger/commands.jsonl').read_text().splitlines()];assert closed[-1]['event']=='finished' and closed[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';assert sha(library.read_bytes())==proof['library_sha256']
plan_path=methods/'evidence/kv-delta-model-source-handoff.json';plan=json.loads(plan_path.read_bytes());staged=results/'kv-delta-model-source.zip'
assert sha(Path(__file__).read_bytes())==plan['runner_sha256']
assert sha(staged.read_bytes())==plan['source_zip_sha256'] and plan['default_off_archive_sha256']==proof['archive_sha256']
build=json.loads((results/'native-kv-delta-build-attempt1/manifest.json').read_bytes());assert build['library_sha256']==proof['library_sha256'] and build['exit_code']==0
for n,h in previous['source_sha256'].items():assert sha((repo/n).read_bytes())==h,('Default-off source drift',n)
candidate={};old={}
with zipfile.ZipFile(staged) as z:
    assert z.testzip() is None and set(z.namelist())==set(plan['candidate_source_sha256'])
    for n,h in plan['candidate_source_sha256'].items():
        candidate[n]=z.read(n);assert sha(candidate[n])==h
        p=repo/n
        if p.exists():
            old[n]=p.read_bytes()
            assert sha(old[n].replace(b'\r\n',b'\n')) in (plan['base_normalized_sha256'].get(n),sha(candidate[n].replace(b'\r\n',b'\n'))),('Unreviewed source drift',n)
        else:assert n not in plan['required_existing_files'],('Missing original runtime source',n)
    cpu=json.loads(candidate['test2026/cuda/qwen38_5090/evidence/kv-delta-model-cpu-tests.json']);assert cpu['status']=='compact_model_cpu_contracts_passed' and cpu['exit_code']==0 and cpu['test_count']==27
    for n,h in cpu['source_sha256'].items():assert sha(candidate[n])==h
for n,h in build['candidate_source_sha256'].items():assert sha((repo/n).read_bytes())==h
scratch_path=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(scratch_path.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root)
assert shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();name='kv-delta-full-b4-t4-attempt1';out=results/name;assert not out.exists();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
for n,b in old.items():
    p=out/'original'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
workflow={'status':'installing_unqualified_candidate','library_sha256':proof['library_sha256'],'parent_archive_sha256':proof['archive_sha256'],
          'source_plan_sha256':sha(plan_path.read_bytes()),'candidate_source_sha256':plan['candidate_source_sha256'],'original_source_sha256':{n:sha(b) for n,b in old.items()},
          'unchanged_native_source_sha256':build['candidate_source_sha256'],'unchanged_parent_source_sha256':{n:h for n,h in previous['source_sha256'].items() if n not in candidate},
          'kv_delta_state':True,'shared_output_storage':True,'cleanup_finished':False,'scope':'Compact B4/T4 full-model correctness and final-write guard qualification only. Retained same-DLL default-off state/IDs/hidden; no stream, HTTP, actual4K, broad-quality or speed admission.'}
path=out/'workflow.json';code=1
def save():path.write_text(json.dumps(workflow,indent=2),encoding='utf8')
save()
try:
    for n,b in candidate.items():
        p=repo/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);assert sha(p.read_bytes())==plan['candidate_source_sha256'][n]
    launcher=root/'test-kv-delta-full-b4-t4.bat';launcher_source=methods/'test_kv_delta_full_b4_t4.bat'
    assert not launcher.exists();launcher.write_bytes(launcher_source.read_bytes())
    workflow['status']='running';workflow['launcher_sha256']=sha(launcher.read_bytes());save()
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],out/'ledger','correctness','compact-canonical-KV-full-model-and-final-write-guards',
        [library,launcher,proof_path,plan_path,scratch_path,Path(__file__)]+[repo/n for n in candidate]+[repo/n for n in build['candidate_source_sha256']])
    if code:raise RuntimeError('Compact full-model qualification failed')
    workflow['status']='compact_model_finished_needs_independent_replay'
except BaseException as error:
    workflow.update(status='failed',error=repr(error))
    if code==0:code=1
    raise
finally:
    workflow['exit_code']=code;workflow['library_unchanged']=sha(library.read_bytes())==proof['library_sha256']
    workflow['source_after_sha256']={n:sha((repo/n).read_bytes()) for n in candidate if (repo/n).exists()}
    workflow['native_after_sha256']={n:sha((repo/n).read_bytes()) for n in build['candidate_source_sha256']}
    workflow['parent_source_after_sha256']={n:sha((repo/n).read_bytes()) for n in workflow['unchanged_parent_source_sha256']};save()
    idle();retired=[];resolved=cache.resolve(strict=True);assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root)
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
    workflow['retired_private_engine_files']=retired;workflow['cleanup_finished']=True;save()
print(json.dumps(workflow,indent=2));raise SystemExit(code)
