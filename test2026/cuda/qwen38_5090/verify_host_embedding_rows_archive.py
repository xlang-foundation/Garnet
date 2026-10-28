"""Independently replay archived native embedding rows and build/source lineage."""
from pathlib import Path,PurePosixPath
import hashlib,json,sys,tempfile,zipfile
from verify_host_embedding_rows import verify
out=Path(sys.argv[1]);repo=Path(sys.argv[2]);sha=lambda raw:hashlib.sha256(raw).hexdigest()
archive=out/'host-embedding-rows-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence=json.loads(z.read('evidence-manifest.json'))
    assert len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for name,digest in evidence['files'].items():
        p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and ':' not in name
        assert sha(z.read(name))==digest
    for name,digest in evidence['linked_archives'].items():assert sha((out/name).read_bytes())==digest
    doc=lambda name:json.loads(z.read(name))
    base='results/host-embedding-rows-attempt1/';bbase='results/native-host-embedding-build-attempt1/'
    build=doc(bbase+'manifest.json');workflow=doc(base+'workflow.json')
    assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
    assert build['library_sha256']==workflow['library_sha256']==evidence['library_sha256']
    assert build['old_library_sha256']==evidence['original_library_sha256']!=evidence['library_sha256']
    assert workflow['status']=='completed_needs_independent_replay' and workflow['exit_code']==0
    assert workflow['build_manifest_sha256']==sha(z.read(bbase+'manifest.json'))
    plan=doc('Garnet/test2026/cuda/qwen38_5090/evidence/host-embedding-source-handoff.json')
    assert build['handoff_sha256']==sha(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/host-embedding-source-handoff.json'))
    assert build['original_source_sha256']==plan['original_source_sha256'] and build['candidate_source_sha256']==plan['candidate_source_sha256']
    for name,digest in build['original_source_sha256'].items():assert sha(z.read(bbase+'original/'+name))==digest
    for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
        for name,digest in build[key].items():assert sha(z.read('Garnet/'+name))==digest
    prior_raw=z.read('results/batch-mtp-deferred-http-pilot-attempt1-verified.json');prior=json.loads(prior_raw)
    assert sha(prior_raw)==build['prior_http_proof_sha256']
    assert prior['archive_sha256']==evidence['linked_archives']['batch-mtp-deferred-http-pilot-attempt1-evidence.zip']
    assert prior['library_sha256']==build['old_library_sha256']
    for ledger in (base+'fixture-ledger/commands.jsonl',base+'native-ledger/commands.jsonl',bbase+'build-ledger/commands.jsonl'):
        events=[json.loads(line) for line in z.read(ledger).decode().splitlines()]
        assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
    # Replay uses the caller's checked verifier, never executable code from ZIP.
    name='test2026/cuda/qwen38_5090/verify_host_embedding_rows.py'
    assert sha((repo/name).read_bytes())==sha(z.read('Garnet/'+name))
    scratch=out.resolve().parent/'work/host-embedding-replay-scratch'
    assert scratch.resolve().is_relative_to(out.resolve().parent)
    scratch.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            path=Path(temp);assert path.resolve().is_relative_to(scratch.resolve())
            for name in evidence['files']:
                if not name.startswith(base):continue
                destination=path/name;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(z.read(name))
            result=verify(path/base/'native',path/base/'fixture',evidence['library_sha256'])
    finally:scratch.rmdir()
    resource=doc(base+'resources/resource-monitor.json')
    samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(samples)
    peak=max(sample['used_bytes'] for sample in samples)
    assert peak==resource['sampled_memory']['sampled_peak_adapter_used_bytes']
result.update(status='host_embedding_selected_rows_and_build_evidence_verified',archive_sha256=sha(archive.read_bytes()),files_verified=len(evidence['files']),sampled_peak_adapter_used_bytes=peak,
    scope='Independent selected embedding byte/layout/nonmutation/retained lifecycle and incremental build/source/terminal replay. Not allocation-failure injection, sanitizer, loaded-module identity, full-model/HTTP correctness, process-memory saving or speed admission.')
(out/'host-embedding-rows-attempt1-verified.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
