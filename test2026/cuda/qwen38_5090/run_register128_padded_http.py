"""Matched compact B4/T4 HTTP pilot after closed full-model/stream replay.

All methods must be deployed before launch. During this pilot observe only its
original job handle: no auxiliary GPU/process queries, uploads, or second worker.
"""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';repo=root/'Garnet';methods=repo/'test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
sha=lambda b:hashlib.sha256(b).hexdigest()
stream='register128-padded-stream-attempt1';proof_path=results/(stream+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='compact_register128_padded_stream_replayed' and proof['groups']==13 and proof['requests']==40
assert len(proof['generation_mode_by_id'])==40 and set(proof['generation_mode_by_id'].values())=={'verified_mtp','greedy_target'}
assert sha((results/(stream+'-evidence.zip')).read_bytes())==proof['archive_sha256']
previous=json.loads((results/stream/'workflow.json').read_bytes());assert previous['exit_code']==0 and previous['cleanup_finished']
events=[json.loads(l) for l in (results/stream/'ledger/commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes());assert digest==proof['library_sha256']
sources=previous['source_sha256']
for n,h in sources.items():assert sha((repo/n).read_bytes())==h
for n,h in previous['stream_source_sha256'].items():assert sha((methods/n).read_bytes())==h
full_path=results/'kv-delta-full-b4-t4-attempt1-verified.json';full=json.loads(full_path.read_bytes())
assert full['status']=='compact_kv_delta_full_model_replayed' and full['raw_last_hidden_all_exact'] and full['library_sha256']==digest and sha((results/'kv-delta-full-b4-t4-attempt1-evidence.zip').read_bytes())==full['archive_sha256']
prep_path=methods/'evidence/register128-padded-http-preparation.json';prep=json.loads(prep_path.read_bytes())
assert sha(Path(__file__).read_bytes())==prep['runner_sha256'] and prep['grouping_delay_ms']==10 and not prep['gpu_run']
for n,h in prep['method_sha256'].items():assert sha((methods/n).read_bytes())==h
for n,h in prep['files'].items():assert sha((methods/n).read_bytes())==h
probe=methods/'probe_register128_padded_http.py';assert sha(probe.read_bytes())==prep['probe_sha256']
launcher_source=methods/'test_register128_padded_http.bat';assert sha(launcher_source.read_bytes())==prep['launcher_sha256']
parent='padded-mtp-http-pilot-attempt1';parent_path=results/(parent+'-verified.json');old=json.loads(parent_path.read_bytes());oldarchive=results/(parent+'-evidence.zip')
assert old['status']=='padded_mtp_http_pilot_replayed' and old['grouping_delay_ms']==10 and sha(oldarchive.read_bytes())==old['archive_sha256']==prep['retained_http_archive_sha256']
closed=json.loads((results/parent/'workflow.json').read_bytes());assert closed['exit_code']==0 and closed['cleanup_finished']
closed_events=[json.loads(l) for l in (results/parent/'ledger/commands.jsonl').read_text().splitlines()]
assert closed_events[-1]['event']=='finished' and closed_events[-1]['exit_code']==0
with zipfile.ZipFile(oldarchive) as z:
    text=z.read('Garnet/test2026/cuda/qwen38_5090/probe_padded_mtp_http.py').decode().replace('\r\n','\n')
    for before,after,count in prep['probe_derivation']:assert text.count(before)==count;text=text.replace(before,after)
    assert text.encode()==probe.read_bytes()
    for n,h in prep['files'].items():
        path='Garnet/test2026/cuda/qwen38_5090/'+n
        assert sha(z.read(path))==h
    evidence=json.loads(z.read('evidence-manifest.json'))
    references={n:evidence['linked_archives'][n] for n in ('vllm-graph-linkfix-evidence.zip','sglang-corrected-ipv4-evidence.zip','batch-http-pilot-attempt1-evidence.zip')}
    for n,h in references.items():assert sha((results/n).read_bytes())==h
repair_path=results/'unicode-transport-repair2-verified.json';repair=json.loads(repair_path.read_bytes())
assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython'] and sha((methods/'native_worker_io.py').read_bytes())==repair['repair_source_sha256']
regression=results/'batch-http-serialization-regression.json';assert len(json.loads(regression.read_bytes())['cases'])==2
cache_receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(cache_receipt.read_bytes());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:' and json.loads((cache/'study-owner.json').read_bytes())['study']==str(root) and shutil.disk_usage(cache).free>=96*1024**3
worker_launcher=root/'serve-native-kv-delta-register128-padded-mtp-worker.bat';assert worker_launcher.read_bytes()==(methods/'serve_native_kv_delta_register128_padded_mtp_worker.bat').read_bytes()
def idle():assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
state_path=results/'register128-padded-state-attempt1-verified.json';state=json.loads(state_path.read_bytes())
assert state['status']=='compact_register128_padded_state_replayed' and state['real_state_all_exact'] and len(state['cases'])==12
assert state['raw_ordinary_mtp_hidden_arrays']==24 and state['reset_hidden_arrays']==12 and state['full_four_default_off_controls']==2 and state['register_delta_128'] is True and state['raw_hidden_register_off_controls']==38
state_archive=results/'register128-padded-state-attempt1-evidence.zip';assert sha(state_archive.read_bytes())==state['archive_sha256']
assert state['candidate_worker_sha256']==sha((methods/'native_kv_delta_register128_padded_mtp_stream_worker.py').read_bytes())
config=root/'models/Qwen3.8-27B-NVFP4/config.json';config_sha=sha(config.read_bytes())
with zipfile.ZipFile(state_archive) as z:assert sha(z.read('models/Qwen3.8-27B-NVFP4/config.json'))==config_sha
idle();out=results/'register128-padded-http-pilot-attempt1';assert not out.exists();out.mkdir()
cpu=subprocess.run([sys.executable,'-m','unittest','test_native_batch_bridge','test_native_http_bridge','test_benchmark_stream','test_compare_http_runs'],cwd=methods,capture_output=True,text=True)
(out/'qpc-bridge-cpu-tests.json').write_text(json.dumps({'exit_code':cpu.returncode,'stdout':cpu.stdout,'stderr':cpu.stderr,'source_sha256':prep['cpu_sources']},indent=2),encoding='utf8')
assert cpu.returncode==0 and 'Ran 21 tests' in cpu.stderr
for n,h in prep['cpu_sources'].items():assert sha((methods/n).read_bytes())==h
for n,h in prep['cpu_results'].items():assert sha((results/n).read_bytes())==h
before={p.name for p in cache.iterdir()};started=time.time()
launcher=root/'test-register128-padded-http.bat';assert not launcher.exists();launcher.write_bytes(launcher_source.read_bytes())
workflow={'status':'running','library_sha256':digest,'padded_state_archive_sha256':state['archive_sha256'],'model_config_sha256':config_sha,'stream_archive_sha256':proof['archive_sha256'],'full_model_archive_sha256':full['archive_sha256'],
          'parent_http_archive_sha256':old['archive_sha256'],'grouping_clock':'perf_counter','reference_archives':references,'source_sha256':sources,'stream_source_sha256':previous['stream_source_sha256'],
          'http_helper_sha256':prep['files'],'preparation_sha256':sha(prep_path.read_bytes()),'grouping_delay_ms':10,'auxiliary_observer_during_timing':False,'cleanup_finished':False,
          'scope':'One register128 private-padding B4/T4 MTP pilot: unchanged seed3 short text/image C1/2/4/8,8requests/cohort,128outputs,2warmups and10ms grouping; HTTP ID/SSE/serialization gates before timing. Original references untuned; actual4K, broad quality and clean repeats pending. No automatic2x admission.'}
path=out/'workflow.json';code=1
def save():path.write_text(json.dumps(workflow,indent=2),encoding='utf8')
save()
try:
    code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],out/'ledger','correctness','compact-KV-HTTP-quality-and-matched-pilot',
        [library,config,state_path,launcher,worker_launcher,probe,proof_path,full_path,parent_path,prep_path,repair_path,regression,cache_receipt,Path(__file__)]+[repo/n for n in sources]+[methods/n for n in previous['stream_source_sha256']]+[methods/n for n in prep['files']])
    if code:raise RuntimeError('Compact HTTP pilot failed')
    workflow['status']='register128_http_finished_needs_independent_replay'
except BaseException as error:
    workflow.update(status='failed',error=repr(error))
    if code==0:code=1
    raise
finally:
    workflow['model_config_after_sha256']=sha(config.read_bytes());workflow['exit_code']=code;workflow['library_unchanged']=sha(library.read_bytes())==digest
    workflow['source_after_sha256']={n:sha((repo/n).read_bytes()) for n in sources};workflow['stream_source_after_sha256']={n:sha((methods/n).read_bytes()) for n in previous['stream_source_sha256']};workflow['http_helper_after_sha256']={n:sha((methods/n).read_bytes()) for n in prep['files']};save()
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
