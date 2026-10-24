"""Unrun B4 T4 host-embedding MTP3 gate after matched B2 HTTP closure."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time,zipfile
root=Path("D:/CantorAI/5090-study-20261023");results=root/"results"
methods=root/"Garnet/test2026/cuda/qwen38_5090";sys.path.insert(0,str(methods))
from run_logged import run
mode='host'
name='shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1'
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
policy_path=methods/'evidence/mtp-b4-t4-builder256-source-policy.json';policy=json.loads(policy_path.read_text())
install_path=results/'mtp-b4-builder256-source-receipt.json';install=json.loads(install_path.read_text())
assert install['status']=='validation_only_policy_installed_unqualified' and install['library_sha256']==digest
assert install['source_policy_sha256']==sha(policy_path.read_bytes())
failed_path=results/'shared-mtp3-full-b4-t4-host-embedding-attempt1-verified.json';failed=json.loads(failed_path.read_text())
assert failed['status']=='host_embedding_mtp3_full_b4_t4_failed_evidence_verified' and not failed['model_gate_passed'] and not failed['comparisons']
assert sha((results/'shared-mtp3-full-b4-t4-host-embedding-attempt1-evidence.zip').read_bytes())==failed['archive_sha256']==policy['prior_failed_archive_sha256']
assert sha(failed_path.read_bytes())==install['prior_failed_proof_sha256']
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
        old=source.read('Garnet/'+path);current=(root/'Garnet'/path).read_bytes()
        if path in policy['files']:
            rule=policy['files'][path];assert sha(old)==rule['old_sha256'] and sha(current)==rule['new_sha256']
            text=old.decode().replace('\r\n','\n');assert text.count(rule['old_validation'])==1
            assert text.replace(rule['old_validation'],rule['new_validation'],1)==current.decode().replace('\r\n','\n')
        else:assert old==current
    old_state=z.read('results/shared-mtp3-full-b2-deferred-attempt1/native/report.json')
    assert old_state==(results/'shared-mtp3-full-b2-deferred-attempt1/native/report.json').read_bytes()
expected_workspace_sources={path:sha((root/'Garnet'/path).read_bytes()) for path in core}
if mode=='host':
    device_path=results/'shared-mtp3-full-b2-host-embedding-device-attempt1-verified.json'
    device=json.loads(device_path.read_text())
    assert device['status']=='host_embedding_full_b2_device_regression_verified' and device['library_sha256']==digest
    assert device['model_gate_passed'] and device['strict_target_state_hash_gate_passed'] and device['retained_old_library_state_exact']
    assert sha((results/'shared-mtp3-full-b2-host-embedding-device-attempt1-evidence.zip').read_bytes())==device['archive_sha256']
host_path=results/'shared-mtp3-full-b2-host-embedding-host-attempt1-verified.json';host=json.loads(host_path.read_text())
assert host['status']=='host_embedding_full_b2_host_regression_verified' and host['library_sha256']==digest
assert host['model_gate_passed'] and host['strict_target_state_hash_gate_passed'] and host['retained_old_library_state_exact']
assert sha((results/'shared-mtp3-full-b2-host-embedding-host-attempt1-evidence.zip').read_bytes())==host['archive_sha256']
latest_path=results/'batch-mtp-host-embedding-http-pilot-attempt1-verified.json';latest=json.loads(latest_path.read_text())
assert latest['status']=='batch_mtp_host_embedding_http_pilot_independently_verified' and latest['library_sha256']==digest and latest['host_embedding_lookup']
assert latest['measured_sse_worker_pairs']==64 and latest['serialization_regression_captures']==4
assert sha((results/'batch-mtp-host-embedding-http-pilot-attempt1-evidence.zip').read_bytes())==latest['archive_sha256']
latest_events=[json.loads(line) for line in (results/'batch-mtp-host-embedding-http-pilot-attempt1/ledger/commands.jsonl').read_text().splitlines()]
assert latest_events[-1]['event']=='finished' and latest_events[-1]['exit_code']==0
old_b4_path=results/'shared-full-b4-t4-attempt1-verified.json';old_b4=json.loads(old_b4_path.read_text())
assert old_b4['status']=='shared_full_b4_t4_id_gate_verified' and old_b4['model_gate_passed']
assert old_b4['library_sha256']==build['old_library_sha256']
assert sha((results/'shared-full-b4-t4-attempt1-evidence.zip').read_bytes())==old_b4['archive_sha256']
receipt=results/'study-ssd-engine-cache/receipt.json';scratch=json.loads(receipt.read_text());cache=Path(scratch['cache_root'])
assert scratch['status']=='ssd_engine_scratch_prepared' and cache.resolve().drive.upper()=='C:'
assert json.loads((cache/'study-owner.json').read_text())['study']==str(root)
assert shutil.disk_usage(cache).free>=96*1024**3
def idle():
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<1024
idle();out.mkdir();before={p.name for p in cache.iterdir()};started=time.time()
launcher=root/'test-shared-mtp3-full-b4-t4-host-embedding-workspace.bat';probe=methods/'probe_shared_mtp3_full_b4_t4_host_embedding_workspace.py'
workflow={'status':'running','library_sha256':digest,'host_embedding_lookup':mode=='host','mode':mode,'physical_batch':4,'target_chunk':4,'draft_chunk':4,'strict_same_profile_state_required':True,'cross_profile_legacy_state_diagnostic_only':True,
    'probe_sha256':sha(probe.read_bytes()),'workspace_source_sha256':expected_workspace_sources,'source_policy_sha256':sha(policy_path.read_bytes()),'source_install_receipt_sha256':sha(install_path.read_bytes()),
    'retained_state_report_sha256':sha(old_state),'draft_builder_workspace_bytes':256<<20,'deferred_state_commit':True,
    'scope':'Unrun B4 T4/draftT4 MTP3 host/deferred4096 candidate: retained B1 IDs and exact same-profile ordinary/MTP states plus duplicate-pair isolation. Old B2 T16 state hashes diagnostic across changed geometry. Full state arrays absent; raw last-hidden retained. Not stream/HTTP/memory fit or performance admission.'}
(out/'workflow.json').write_text(json.dumps(workflow,indent=2),encoding='utf-8')
code=run([sys.executable,str(methods/'monitor_native_probe.py'),'--study',str(root),'--output',str(out/'resources'),'--','cmd.exe','/d','/c',str(launcher),str(out/'native'),str(cache),mode],out/'ledger','correctness','host-embedding-b4-t4-mtp3-state-gate',
    [library,launcher,probe,policy_path,install_path,failed_path,build_path,row_proof_path,prior_path,http_proof_path,host_path,latest_path,old_b4_path,receipt,Path(__file__)]+[root/'Garnet'/path for path in core]+[root/'Garnet'/path for path in build['candidate_source_sha256']])
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
