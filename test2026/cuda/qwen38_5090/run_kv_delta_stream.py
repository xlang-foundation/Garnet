"""Gate compact 13-group/40-request streaming after full-model replay."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='kv-delta-full-b4-t4-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='compact_kv_delta_full_model_replayed' and proof['raw_last_hidden_arrays_compared']==8 and proof['raw_last_hidden_all_exact']
assert sha((methods/'verify_kv_delta_full_b4_t4.py').read_bytes())==proof['replay_verifier_sha256']
assert len(proof['cases'])==4 and all(c['same_DLL_default_off_exact'] and c['same_profile_state_exact'] and c['ids_exact'] for c in proof['cases'])
assert proof['negative_commit_gate']['status']=='passed' and proof['negative_commit_gate']['metadata_rejections']==136
archive=results/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
previous=json.loads((results/parent/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
closed=[json.loads(l) for l in (results/parent/'ledger/commands.jsonl').read_text().splitlines()];assert closed[-1]['event']=='finished' and closed[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==proof['library_sha256']
sources={**previous['candidate_source_sha256'],**previous['unchanged_parent_source_sha256'],**previous['unchanged_native_source_sha256']}
for n,h in sources.items():assert sha((repo/n).read_bytes())==h,('Source drift',n)
prep_path=methods/'evidence/kv-delta-stream-preparation.json';prep=json.loads(prep_path.read_bytes())
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
assert sha(Path(__file__).read_bytes())==prep['runner_sha256'] and prep['groups']==13 and prep['requests']==40
old='b4-host-joined-mtp-stream-attempt1';oldproof=json.loads((results/(old+'-verified.json')).read_bytes())
assert oldproof['status']=='b4_host_joined_mtp_stream_replayed' and oldproof['exact_ids_and_prompt_ids'] and oldproof['groups']==13 and oldproof['requests']==40
assert sha((results/(old+'-evidence.zip')).read_bytes())==oldproof['archive_sha256']==prep['retained_stream_archive_sha256']
with zipfile.ZipFile(results/(old+'-evidence.zip')) as z:
    for key,source,target in (('worker_derivation','native_b4_host_joined_mtp_stream_worker.py','native_kv_delta_stream_worker.py'),('probe_derivation','probe_b4_host_joined_mtp_stream.py','probe_kv_delta_stream.py')):
        text=z.read('Garnet/test2026/cuda/qwen38_5090/'+source).decode().replace('\r\n','\n')
        for before,after,count in prep[key]:assert text.count(before)==count;text=text.replace(before,after)
        assert text.encode()==(methods/target).read_bytes()
repair_path=results/'unicode-transport-repair2-verified.json';repair=json.loads(repair_path.read_bytes())
assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython'] and sha((methods/'native_worker_io.py').read_bytes())==repair['repair_source_sha256']
scratch_path=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(scratch_path.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out=results/'kv-delta-stream-attempt1';assert not out.exists();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
for source,target in (('serve_native_kv_delta_worker.bat','serve-native-kv-delta-worker.bat'),('test_kv_delta_stream.bat','test-kv-delta-stream.bat')):
    p=root/target;assert not p.exists();p.write_bytes((methods/source).read_bytes())
workflow={'status':'running','library_sha256':digest,'full_model_archive_sha256':proof['archive_sha256'],'retained_stream_archive_sha256':oldproof['archive_sha256'],
          'source_sha256':sources,'stream_source_sha256':prep['files'],'preparation_sha256':sha(prep_path.read_bytes()),'cleanup_finished':False,
          'scope':'Compact B4/T4 host/deferred MTP3 streaming:13groups/40requests, unchanged teacher/wire policy, full-four MTP and partial/unequal greedy fallback. No HTTP, actual4K or performance admission.'}
path=out/'workflow.json';code=1
def save():path.write_text(json.dumps(workflow,indent=2),encoding='utf8')
save()
try:
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(root/'test-kv-delta-stream.bat'),str(out/'native'),str(cache)],out/'ledger','correctness','compact-state-stream-protocol-and-row-isolation',
        [library,proof_path,prep_path,repair_path,scratch_path,Path(__file__),root/'test-kv-delta-stream.bat',root/'serve-native-kv-delta-worker.bat']+[repo/n for n in sources]+[methods/n for n in prep['files']])
    if code:raise RuntimeError('Compact streaming correctness failed')
    workflow['status']='compact_stream_finished_needs_independent_replay'
except BaseException as error:
    workflow.update(status='failed',error=repr(error))
    if code==0:code=1
    raise
finally:
    workflow['exit_code']=code;workflow['library_unchanged']=sha(library.read_bytes())==digest
    workflow['source_after_sha256']={n:sha((repo/n).read_bytes()) for n in sources};workflow['stream_source_after_sha256']={n:sha((methods/n).read_bytes()) for n in prep['files']};save()
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
