"""Fresh full-model regression of the new DLL with canonical KV mode disabled."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
name='native-kv-delta-default-off-attempt1';out=results/name;assert not out.exists()
build_path=results/'native-kv-delta-build-attempt1/manifest.json';build=json.loads(build_path.read_bytes())
assert build['status']=='native_kv_delta_build_finished_unqualified' and build['exit_code']==0
build_events=[json.loads(l) for l in (build_path.parent/'build-ledger/commands.jsonl').read_text().splitlines()]
assert build_events[-1]['event']=='finished' and build_events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==build['library_sha256']==sha(Path(build['retained_candidate_library']).read_bytes())
parent='shared-mtp3-full-b4-t4-host-joined-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='b4_host_joined_mtp_correctness_replayed' and proof['model_gate_passed'] and proof['raw_last_hidden_all_exact']
assert build['old_library_sha256']==proof['library_sha256']
archive=results/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0
previous_events=[json.loads(l) for l in (results/parent/'ledger/commands.jsonl').read_text().splitlines()]
assert previous_events[-1]['event']=='finished' and previous_events[-1]['exit_code']==0
host=json.loads((results/'native-host-embedding-build-attempt1/manifest.json').read_bytes())
sources={**previous['workspace_source_sha256'],**host['candidate_source_sha256'],**host['unchanged_policy_source_sha256'],**build['candidate_source_sha256']}
for n,h in sources.items():assert sha((repo/n).read_bytes())==h,('Source drift',n)
probe=methods/'probe_b4_host_joined_mtp.py';launcher=root/'test-b4-host-joined-mtp.bat'
with zipfile.ZipFile(archive) as z:
    assert probe.read_bytes()==z.read('Garnet/test2026/cuda/qwen38_5090/'+probe.name) and launcher.read_bytes()==z.read(launcher.name)
repair_path=results/'unicode-transport-repair2-verified.json';repair=json.loads(repair_path.read_bytes())
assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
assert sha((methods/'native_worker_io.py').read_bytes())==repair['repair_source_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
workflow={'status':'running','library_sha256':digest,'build_manifest_sha256':sha(build_path.read_bytes()),'parent_archive_sha256':proof['archive_sha256'],
          'source_sha256':sources,'probe_sha256':sha(probe.read_bytes()),'canonical_kv_overlay_enabled':False,'kv_delta_commit_enabled':False,'cleanup_finished':False,
          'scope':'New native DLL, unchanged joined-host B4/T4 full-model probe and original full-cache model path. Retained IDs/state/hidden regression only; new compact helper/model path and performance unqualified.'}
path=out/'workflow.json';code=1
def save():path.write_text(json.dumps(workflow,indent=2),encoding='utf8')
save()
try:
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],out/'ledger','correctness','new-native-KV-DLL-default-off-full-model-regression',
        [library,launcher,probe,build_path,proof_path,repair_path,receipt,Path(__file__)]+[repo/n for n in sources])
    if code:raise RuntimeError('Default-off full-model regression failed')
    workflow['status']='default_off_finished_needs_independent_replay'
except BaseException as error:
    workflow.update(status='failed',error=repr(error))
    if code==0:code=1
    raise
finally:
    workflow['exit_code']=code;workflow['library_unchanged']=sha(library.read_bytes())==digest
    workflow['source_after_sha256']={n:sha((repo/n).read_bytes()) for n in sources}
    save()
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
    workflow['retired_private_engine_files']=retired;workflow['cleanup_finished']=True;save()
print(json.dumps(workflow,indent=2));raise SystemExit(code)
