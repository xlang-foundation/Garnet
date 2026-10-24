"""Run separate synchronized wall diagnostics after original HTTP closure/archive."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
root=Path('D:/CantorAI/5090-study-20261023');results=root/'results';methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,str(methods))
from run_logged import run
out=results/'b4-mtp-state-wall-trace-attempt1';assert not out.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
build_path=results/'native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_text())
assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes())
assert digest==build['library_sha256']==sha(Path(build['retained_candidate_library']).read_bytes())
for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
    for name,want in build[key].items():assert sha((root/'Garnet'/name).read_bytes())==want
proof_path=results/'shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1-verified.json';proof=json.loads(proof_path.read_text())
assert proof['status']=='host_embedding_mtp3_full_b4_t4_state_gate_verified' and proof['library_sha256']==digest
assert proof['model_gate_passed'] and proof['strict_same_profile_target_state_hash_gate_passed'] and proof['raw_last_hidden_arrays_verified']==8 and proof['draft_builder_workspace_bytes']==256<<20
full_archive=results/'shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1-evidence.zip'
assert sha(full_archive.read_bytes())==proof['archive_sha256']
closed=[json.loads(line) for line in (results/'shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1/ledger/commands.jsonl').read_text().splitlines()]
assert closed[-1]['event']=='finished' and closed[-1]['exit_code']==0
full_workflow=json.loads((results/'shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1/workflow.json').read_text())
for name,want in full_workflow['workspace_source_sha256'].items():assert sha((root/'Garnet'/name).read_bytes())==want
repair_path=results/'unicode-transport-repair2-verified.json';repair=json.loads(repair_path.read_text())
assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
assert sha((methods/'native_worker_io.py').read_bytes())==repair['repair_source_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_text());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_text())['study']==str(root)
assert shutil.disk_usage(cache).free>=96*1024**3
def idle():
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
http_name='batch4-t4-mtp-host-embedding-http-pilot-attempt1'
http_workflow=json.loads((results/http_name/'workflow.json').read_text())
assert http_workflow['status'] in ('process_failed','completed_needs_independent_replay')
http_events=[json.loads(line) for line in (results/http_name/'ledger/commands.jsonl').read_text().splitlines()]
assert http_events[-1]['event']=='finished' and http_events[-1]['exit_code']==http_workflow['exit_code']
http_archive=results/(http_name+'-evidence.zip');assert http_archive.is_file()
import zipfile
with zipfile.ZipFile(http_archive) as archived:
    assert archived.testzip() is None
    evidence=json.loads(archived.read('evidence-manifest.json'))
    assert evidence['library_sha256']==digest
    for name,want in evidence['files'].items():assert sha(archived.read(name))==want
    assert archived.read('results/'+http_name+'/workflow.json')==(results/http_name/'workflow.json').read_bytes()
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
launcher=root/'test-b4-mtp-state-wall-trace.bat';probe=methods/'probe_b4_mtp_state_wall_trace.py'
worker=methods/'probe_shared_mtp3_full_b4_t4_host_embedding_workspace.py'
assert sha(worker.read_bytes())==full_workflow['probe_sha256']
workflow={'status':'running','library_sha256':digest,'probe_sha256':sha(probe.read_bytes()),'full_correctness_probe_sha256':sha(worker.read_bytes()),'prior_http_archive_sha256':sha(http_archive.read_bytes()),'host_embedding_lookup':True,
    'workspace_source_sha256':full_workflow['workspace_source_sha256'],'full_host_proof_sha256':sha(proof_path.read_bytes()),
    'scope':'Synchronized nested MTP generate wall diagnostics while rerunning unchanged B4 full-model correctness. Instrumentation perturbs scheduling. Parent intervals include children; not production timing or speedup.'}
(out/'workflow.json').write_text(json.dumps(workflow,indent=2),encoding='utf-8')
code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache)],out/'ledger','correctness','b4-mtp-state-wall-diagnostic',[library,launcher,probe,worker,http_archive,build_path,proof_path,repair_path,receipt,Path(__file__)]+[root/'Garnet'/name for name in full_workflow['workspace_source_sha256']]+[root/'Garnet'/name for name in build['candidate_source_sha256']])
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
