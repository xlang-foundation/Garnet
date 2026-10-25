"""Separate marked CUDA trace after closed and replayed padded HTTP pilot."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
parent='padded-mtp-http-pilot-attempt1';proof_path=results/(parent+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='padded_mtp_http_pilot_replayed' and proof['measured_sse_worker_pairs']==64
assert sha((methods/'verify_padded_mtp_http.py').read_bytes())==proof['replay_verifier_sha256']
archive=results/(parent+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
closed=json.loads((results/parent/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished']
ledger=[json.loads(l) for l in (results/parent/'ledger/commands.jsonl').read_text().splitlines()]
assert ledger[-1]['event']=='finished' and ledger[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==proof['library_sha256']==closed['library_sha256']
config=root/'models/Qwen3.8-27B-NVFP4/config.json';config_sha=sha(config.read_bytes());assert config_sha==closed['model_config_sha256']==closed['model_config_after_sha256']
sources=closed['source_sha256'];helpers={**closed['stream_source_sha256'],**closed['http_helper_sha256']}
for n,h in sources.items():assert sha((repo/n).read_bytes())==h
for n,h in helpers.items():assert sha((methods/n).read_bytes())==h
prep_path=methods/'evidence/kv-delta-decode-cuda-trace-preparation.json';prep=json.loads(prep_path.read_bytes())
assert not prep['gpu_run'] and not prep['performance_admitted'] and prep['case_count']==8
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
assert prep['files'][Path(__file__).name]==sha(Path(__file__).read_bytes())
full='kv-delta-full-b4-t4-attempt1';full_archive=results/(full+'-evidence.zip');full_path=results/(full+'-verified.json');full_proof=json.loads(full_path.read_bytes())
assert full_proof['status']=='compact_kv_delta_full_model_replayed' and full_proof['raw_last_hidden_all_exact']
assert sha(full_archive.read_bytes())==full_proof['archive_sha256']==prep['full_model_archive_sha256']
with zipfile.ZipFile(full_archive) as z:
    text=z.read('Garnet/test2026/cuda/qwen38_5090/probe_kv_delta_full_b4_t4.py').decode().replace('\r\n','\n')
    for before,after,count in prep['probe_derivation']:assert text.count(before)==count;text=text.replace(before,after)
    assert text.encode()==(methods/'probe_kv_delta_warm_full_b4_t4.py').read_bytes()
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
smoke=results/'nsight-cuda-smoke-2026-attempt1-evidence.zip'
assert sha(smoke.read_bytes())=='cc37ade5bd19ba07aa35e93172b88ddcfd2d20e3b38f5ff02cccb1da4a615b9f'
setup_path=results/'nsight-2026-cab-extraction/setup.json';setup=json.loads(setup_path.read_bytes())
with zipfile.ZipFile(smoke) as z:assert z.read('results/nsight-2026-cab-extraction/setup.json')==setup_path.read_bytes()
assert setup['status']=='windows_payload_and_cli_version_verified'
nsys=Path(setup['nsys_path']);tool_sha=sha(nsys.read_bytes())
assert tool_sha==setup['nsys_sha256']=='e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291'
version=subprocess.check_output([str(nsys),'--version'],text=True).strip();assert '2026.5.1.161-265138896106v0' in version
assert shutil.disk_usage(root).free>=5*1024**3
idle();out=results/'kv-delta-decode-cuda-trace-attempt1';assert not out.exists();out.mkdir()
cpu=subprocess.run([sys.executable,'-m','unittest','test_warm_host_trace','test_cuda_trace_windows'],cwd=methods,capture_output=True,text=True)
(out/'cpu-tests.json').write_text(json.dumps({'exit_code':cpu.returncode,'stdout':cpu.stdout,'stderr':cpu.stderr},indent=2),encoding='utf8')
assert cpu.returncode==0 and 'Ran 10 tests' in cpu.stderr
before={p.name for p in cache.iterdir()};started=time.time()
launcher=root/'test-kv-delta-decode-cuda-trace.bat';assert not launcher.exists();launcher.write_bytes((methods/'test_kv_delta_decode_cuda_trace.bat').read_bytes())
workflow={'status':'running','library_sha256':digest,'model_config_sha256':config_sha,'nsys_sha256':tool_sha,'nsys_version':version,'nsight_smoke_archive_sha256':sha(smoke.read_bytes()),'parent_http_archive_sha256':proof['archive_sha256'],'full_model_archive_sha256':full_proof['archive_sha256'],
    'source_sha256':sources,'helper_sha256':helpers,'method_sha256':prep['files'],'preparation_sha256':sha(prep_path.read_bytes()),'cleanup_finished':False,'scope':prep['scope']}
path=out/'workflow.json';code=1
def save():path.write_text(json.dumps(workflow,indent=2),encoding='utf8')
save()
try:
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--',str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','--output='+str(out/'startup-trace'),'cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],
        out/'ledger','correctness','compact-warmed-decode-only-cuda-markers',
        [library,nsys,setup_path,launcher,proof_path,full_path,prep_path,receipt,Path(__file__)]+[repo/n for n in sources]+[methods/n for n in {**helpers,**prep['files']}])
    workflow['profile_exit_code']=code;save()
    if code:raise RuntimeError('Marked compact decode trace failed')
    assert json.loads((out/'native/trace-report.json').read_bytes())['status']=='decode_cuda_trace_correctness_finished_needs_replay'
    code=run([str(nsys),'stats','--report','cuda_gpu_kern_sum,cuda_api_sum','--format','csv',str(out/'startup-trace.nsys-rep')],out/'ledger','correctness','raw-startup-trace-stats-not-decode-ranking',[nsys])
    workflow['stats_exit_code']=code
    if code:raise RuntimeError('Nsight stats/export failed')
    assert (out/'startup-trace.sqlite').is_file()
    workflow['status']='decode_cuda_trace_finished_needs_independent_replay'
except BaseException as failure:
    workflow.update(status='failed',error=repr(failure))
    if code==0:code=1
    raise
finally:
    workflow['model_config_after_sha256']=sha(config.read_bytes());workflow['nsys_after_sha256']=sha(nsys.read_bytes());workflow['exit_code']=code;workflow['library_unchanged']=sha(library.read_bytes())==digest
    workflow['source_after_sha256']={n:sha((repo/n).read_bytes()) for n in sources}
    workflow['helper_after_sha256']={n:sha((methods/n).read_bytes()) for n in helpers}
    workflow['method_after_sha256']={n:sha((methods/n).read_bytes()) for n in prep['files']};save()
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
