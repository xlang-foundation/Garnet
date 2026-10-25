"""Release a candidate timing job only after the owned host diagnosis closes."""
import hashlib
import json
import zipfile


def closed_host_diagnostic(root, digest):
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Actual closed host diagnostic SHA256 required')
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    name = 'native-host-stage-full4k-attempt1'
    archive = root / 'results' / (name + '-evidence.zip')
    assert sha(archive.read_bytes()) == digest
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for path, expected in manifest['files'].items():
            assert sha(z.read(path)) == expected
        w = json.loads(z.read('results/' + name + '/workflow.json'))
    assert w == json.loads((root / 'results' / name / 'workflow.json').read_bytes())
    assert type(w['exit_code']) is int and w['cleanup_finished'] is True and w['library_unchanged'] is True
    assert w['source_sha256'] == w['source_after_sha256'] and w['runtime_dependencies'] == w['runtime_dependencies_after']
    proof_path = root / 'results' / (name + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    status = 'native_host_stage_full4k_replayed' if w['exit_code'] == 0 else 'native_host_stage_full4k_failure_preserved'
    assert proof['archive_sha256'] == digest and proof['status'] == status
    return {'job': name, 'archive_sha256': digest, 'proof_sha256': sha(proof_path.read_bytes()),
            'proof_status': status, 'exit_code': w['exit_code'], 'cleanup_finished': True}


def finish_worker(worker, out):
    """Record the actual owned worker exit, including failed cleanup attempts."""
    worker.close()
    code = worker.process.returncode
    path = out / 'worker-exit.json'
    with path.open('x', encoding='utf8') as stream:
        json.dump({'exit_code': code, 'scope': 'Owned worker actual exit after HTTP shutdown; not latency'}, stream, indent=2)
    assert type(code) is int and code == 0, ('Owned worker did not close cleanly', code)
