"""Closed-job/profiler guards for the explicit full4K CUDA diagnosis."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

JOB = 'native-cuda-phase-full4k-attempt1'
TRIAL = 'prefill16-fixed4k-http-attempt1'
NSYS_SHA256 = 'e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def closed_trial(root, digest):
    if re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('Closed wider-MTP HTTP archive SHA256 required')
    archive = root / 'results' / (TRIAL + '-evidence.zip')
    assert sha(archive) == digest
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, expected in manifest['files'].items():
            with z.open(name) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected
        workflow = json.loads(z.read('results/' + TRIAL + '/workflow.json'))
    assert workflow == json.loads((root / 'results' / TRIAL / 'workflow.json').read_bytes())
    assert type(workflow['exit_code']) is int and workflow['cleanup_finished'] is True
    assert workflow['source_sha256'] == workflow['source_after_sha256']
    assert workflow['runtime_dependencies'] == workflow['runtime_dependencies_after']
    assert workflow['library_unchanged'] is True
    proof_path = root / 'results' / (TRIAL + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    status = 'prefill16_fixed4k_http_replayed' if workflow['exit_code'] == 0 else 'prefill16_fixed4k_http_failure_preserved'
    assert proof['status'] == status and proof['archive_sha256'] == digest
    return {'job': TRIAL, 'archive_sha256': digest, 'proof_sha256': sha(proof_path),
            'proof_status': status, 'exit_code': workflow['exit_code'], 'cleanup_finished': True}


def profiler(root):
    setup_path = root / 'results/nsight-2026-cab-extraction/setup.json'
    setup = json.loads(setup_path.read_bytes())
    path = root / 'tools/n26/target-windows-x64/nsys.exe'
    assert setup['status'] == 'windows_payload_and_cli_version_verified'
    assert Path(setup['nsys_path']).resolve() == path.resolve()
    assert setup['nsys_sha256'] == NSYS_SHA256 == sha(path)
    assert setup['nsys_authenticode']['Status'] == 'Valid'
    return {'setup_path': setup_path.relative_to(root).as_posix(), 'setup_sha256': sha(setup_path),
            'path': path.relative_to(root).as_posix(), 'sha256': NSYS_SHA256, 'version': setup['version']}
