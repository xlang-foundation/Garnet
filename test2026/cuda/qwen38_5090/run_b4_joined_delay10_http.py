"""Gate matched host-embedding B4/T4 HTTP timing after independent streaming replay."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
out=results/'b4-joined-delay10-http-pilot-attempt1';assert not out.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
build_path=results/'native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_text())
assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes())
assert digest==build['library_sha256']==sha(Path(build['retained_candidate_library']).read_bytes())
for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
    for name,want in build[key].items():assert sha((root/'Garnet'/name).read_bytes())==want
proof_path=results/'shared-mtp3-full-b4-t4-host-joined-attempt1-verified.json';proof=json.loads(proof_path.read_text())
assert proof['status']=='b4_host_joined_mtp_correctness_replayed' and proof['library_sha256']==digest
assert proof['model_gate_passed'] and len(proof['cases'])==4 and all(c['retained_unjoined_exact'] and c['same_profile_state_exact'] and c['ids_exact'] for c in proof['cases']) and proof['raw_last_hidden_arrays_compared']==8 and proof['raw_last_hidden_all_exact']
full_archive=results/'shared-mtp3-full-b4-t4-host-joined-attempt1-evidence.zip'
assert sha(full_archive.read_bytes())==proof['archive_sha256']
closed=[json.loads(line) for line in (results/'shared-mtp3-full-b4-t4-host-joined-attempt1/ledger/commands.jsonl').read_text().splitlines()]
assert closed[-1]['event']=='finished' and closed[-1]['exit_code']==0
full_workflow=json.loads((results/'shared-mtp3-full-b4-t4-host-joined-attempt1/workflow.json').read_text())
for name,want in full_workflow['workspace_source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==want
stream_path=results/'b4-host-joined-mtp-stream-attempt1-verified.json';stream=json.loads(stream_path.read_text())
assert stream['status']=='b4_host_joined_mtp_stream_replayed' and stream['library_sha256']==digest and stream['host_embedding_lookup']
assert sha((results/'b4-host-joined-mtp-stream-attempt1-evidence.zip').read_bytes())==stream['archive_sha256']
assert stream['groups']==13 and stream['requests']==40 and stream['mtp_and_greedy_fallback_modes_verified']
stream_events=[json.loads(line) for line in (results/'b4-host-joined-mtp-stream-attempt1/ledger/commands.jsonl').read_text().splitlines()]
assert stream_events[-1]['event']=='finished' and stream_events[-1]['exit_code']==0
negative_path=results/'batch-http-pilot-attempt1-failure-verified.json';negative=json.loads(negative_path.read_text())
assert negative['status']=='batch_http_failed_pilot_evidence_verified' and not negative['full_run_passed']
regression_path=results/'batch-http-serialization-regression.json';regression=json.loads(regression_path.read_text())
assert regression['source_archive_sha256']==negative['archive_sha256'] and len(regression['cases'])==2
repair_path=results/'unicode-transport-repair2-verified.json';repair=json.loads(repair_path.read_text())
assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
assert sha((methods/'native_worker_io.py').read_bytes())==repair['repair_source_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_text());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_text())['study']==str(root)
assert shutil.disk_usage(cache).free>=96*1024**3
def idle():
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
policy_path=methods/'evidence/b4-joined-mlp-source-policy.json';policy=json.loads(policy_path.read_bytes())
assert sha((root/'Garnet'/policy['file']).read_bytes())==policy['new_sha256']
assert stream['joined_mlp_projections']
parent_name='b4-host-joined-mtp-http-pilot-attempt1'
parent_path=results/(parent_name+'-verified.json');parent=json.loads(parent_path.read_bytes())
assert parent['status']=='b4_host_joined_mtp_http_pilot_replayed' and parent['joined_mlp_projections'] and parent['library_sha256']==digest
parent_archive=results/(parent_name+'-evidence.zip');assert sha(parent_archive.read_bytes())==parent['archive_sha256']
parent_events=[json.loads(line) for line in (results/parent_name/'ledger/commands.jsonl').read_text().splitlines()]
assert parent_events[-1]['event']=='finished' and parent_events[-1]['exit_code']==0
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
launcher=root/'test-b4-joined-delay10-http.bat';probe=methods/'probe_b4_joined_delay10_http.py'
worker=methods/'native_b4_host_joined_mtp_stream_worker.py';worker_launcher=root/'serve-native-b4-host-joined-mtp-worker.bat'
workflow={'status':'running','library_sha256':digest,'probe_sha256':sha(probe.read_bytes()),'worker_sha256':sha(worker.read_bytes()),'host_embedding_lookup':True,'grouping_delay_ms':10,'parent_pilot_proof_sha256':sha(parent_path.read_bytes()),'joined_mlp_projections':True,'joined_policy_sha256':sha(policy_path.read_bytes()),
    'workspace_source_sha256':full_workflow['workspace_source_sha256'],'full_host_proof_sha256':sha(proof_path.read_bytes()),
    'scope':'B4 T4/draftT4 MTP3 host embedding: HTTP ID/SSE/serialization checks before matched exploratory C1/2/4/8 timing. No 2x admission without independent raw replay.'}
(out/'workflow.json').write_text(json.dumps(workflow,indent=2),encoding='utf-8')
code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],out/'ledger','correctness','host-embedding-b4-t4-mtp3-http-pilot',[library,launcher,probe,worker,worker_launcher,parent_path,parent_archive,policy_path,root/'Garnet'/policy['file'],build_path,proof_path,stream_path,negative_path,regression_path,repair_path,receipt,Path(__file__),methods/'native_worker_io.py',methods/'native_batch_http_bridge.py',methods/'native_http_bridge.py',methods/'benchmark_stream.py',methods/'metrics.py']+[root/'Garnet'/name for name in full_workflow['workspace_source_sha256']]+[root/'Garnet'/name for name in build['candidate_source_sha256']])
workflow['exit_code']=code;workflow['status']='process_failed' if code else 'completed_needs_independent_replay'
try:
    idle()
    assert hashlib.sha256(library.read_bytes()).hexdigest() == digest
    retired = []; resolved = cache.resolve(strict=True)
    assert json.loads((cache / "study-owner.json").read_text())["study"] == str(root)
    for path in cache.iterdir():
        if path.name in before or not path.name.startswith("garnet-cuda-graph-"): continue
        st = path.lstat(); target = path.resolve(strict=True)
        assert not (getattr(st, "st_file_attributes", 0) & 0x400)
        assert target.parent == resolved and st.st_ctime >= started - 2
        entries = list(target.iterdir())
        assert all(p.name in ("model.engine", "model.engine.tmp") for p in entries)
        for p in entries:
            ps = p.lstat()
            assert p.is_file() and not (getattr(ps, "st_file_attributes", 0) & 0x400)
            assert p.resolve(strict=True).parent == target
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
            retired.append({"path": str(p), "bytes": ps.st_size, "sha256": h.hexdigest()})
        # Guarded, nonrecursive deletion of this closed run's private engine cache.
        for p in entries: p.unlink()
        target.rmdir()
    workflow["retired_private_engine_files"] = retired
finally:
    (out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf8")
print(json.dumps(workflow, indent=2))
raise SystemExit(code)
