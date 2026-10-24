"""New-DLL default/host B2 full-model state regression after selected-row replay."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path("D:/CantorAI/5090-study-20261023");results=root/"results"
methods=root/"Garnet/test2026/cuda/qwen38_5090";sys.path.insert(0,str(methods))
from run_logged import run
mode=sys.argv[1];assert mode in ('device','host')
name='shared-mtp3-full-b2-host-embedding-'+mode+'-attempt1'
out=results/name;assert not out.exists()
sha=lambda raw:hashlib.sha256(raw).hexdigest()
build_path=results/'native-host-embedding-build-attempt1/manifest.json';build=json.loads(build_path.read_text())
assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
library=root/'build/bin/garnet.dll';digest=sha(library.read_bytes())
assert digest==build['library_sha256']==sha(Path(build['retained_candidate_library']).read_bytes())
for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
    for path,want in build[key].items():assert sha((root/'Garnet'/path).read_bytes())==want
row_proof_path=results/'host-embedding-rows-attempt1-verified.json';row_proof=json.loads(row_proof_path.read_text())
assert row_proof['status']=='host_embedding_selected_rows_and_build_evidence_verified'
assert row_proof['library_sha256']==digest and row_proof['raw_arrays_verified']==226 and row_proof['rejections_verified']==22
row_archive=results/'host-embedding-rows-attempt1-evidence.zip'
assert sha(row_archive.read_bytes())==row_proof['archive_sha256']
row_events=[json.loads(line) for line in (results/'host-embedding-rows-attempt1/native-ledger/commands.jsonl').read_text().splitlines()]
assert row_events[-1]['event']=='finished' and row_events[-1]['exit_code']==0
prior_path=results/'shared-mtp3-full-b2-deferred-attempt1-verified.json';prior=json.loads(prior_path.read_text())
assert prior['status']=='shared_mtp3_full_b2_deferred_id_state_hash_gate_verified' and prior['model_gate_passed'] and prior['strict_target_state_hash_gate_passed']
assert prior['library_sha256']==build['old_library_sha256']
prior_archive=results/'shared-mtp3-full-b2-deferred-attempt1-evidence.zip'
assert sha(prior_archive.read_bytes())==prior['archive_sha256']
core=['xModel/qwen35/vl_0_8b/'+n for n in ('cuda_packed_program.py','cuda_speculative.py','cuda_mtp.py','cuda_packed_replay.py','cuda_verify.py')]+['xModel/qwen38/vl_27b/model.py','test2026/cuda/qwen38_5090/native_prompt.py']
http_proof_path=results/'batch-mtp-deferred-http-pilot-attempt1-verified.json'
http_proof=json.loads(http_proof_path.read_text())
assert sha(http_proof_path.read_bytes())==build['prior_http_proof_sha256']
http_archive=results/'batch-mtp-deferred-http-pilot-attempt1-evidence.zip'
assert sha(http_archive.read_bytes())==http_proof['archive_sha256']
with zipfile.ZipFile(prior_archive) as z, zipfile.ZipFile(http_archive) as hz:
    assert z.testzip() is None
    for path in core:
        source=hz if path.endswith('/native_prompt.py') else z
        assert source.read('Garnet/'+path)==(root/'Garnet'/path).read_bytes()
    old_state=z.read('results/shared-mtp3-full-b2-deferred-attempt1/native/report.json')
    assert old_state==(results/'shared-mtp3-full-b2-deferred-attempt1/native/report.json').read_bytes()
expected_workspace_sources={path:sha((root/'Garnet'/path).read_bytes()) for path in core}
if mode=='host':
    device_path=results/'shared-mtp3-full-b2-host-embedding-device-attempt1-verified.json'
    device=json.loads(device_path.read_text())
    assert device['status']=='host_embedding_full_b2_device_regression_verified' and device['library_sha256']==digest
    assert device['model_gate_passed'] and device['strict_target_state_hash_gate_passed'] and device['retained_old_library_state_exact']
    assert sha((results/'shared-mtp3-full-b2-host-embedding-device-attempt1-evidence.zip').read_bytes())==device['archive_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_text());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_text())['study']==str(root)
assert shutil.disk_usage(cache).free>=96*1024**3
def idle():
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
launcher=root/'test-shared-mtp3-full-b2-host-embedding.bat';probe=methods/'probe_shared_mtp3_full_b2_host_embedding.py'
workflow={'status':'running','library_sha256':digest,'host_embedding_lookup':mode=='host','mode':mode,
    'probe_sha256':sha(probe.read_bytes()),'workspace_source_sha256':expected_workspace_sources,
    'retained_state_report_sha256':sha(old_state),'draft_builder_workspace_bytes':128<<20,'deferred_state_commit':True,
    'scope':'B2 T16/draftT4 MTP3 4096 capacity, host feature off/on; retained B1 IDs, full ordinary/MTP state hash witnesses and old-DLL witnesses. Not HTTP/broad quality/performance.'}
(out/'workflow.json').write_text(json.dumps(workflow,indent=2),encoding='utf-8')
code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache),mode],out/'ledger','correctness','host-embedding-full-b2-'+mode+'-state-regression',
    [library,launcher,probe,build_path,row_proof_path,prior_path,http_proof_path,receipt,Path(__file__)]+[root/'Garnet'/path for path in core]+[root/'Garnet'/path for path in build['candidate_source_sha256']])
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
