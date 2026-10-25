"""Require the actual wider-MTP full4K proof before an explicit HTTP worker loads."""
from pathlib import Path
import hashlib
import json
import zipfile

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def require(root, mode, library_sha256):
    if mode != 'mtp':
        raise ValueError('Explicit prefill16 HTTP worker requires MTP mode')
    name = 'mtp-prefill16-full4k-admission-attempt1'
    proof_path = root / 'results' / (name + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    assert proof['status'] == 'mtp_prefill16_full4k_target_state_replayed'
    assert proof['actual_full4k_correctness_qualified'] is True and proof['actual_full4k_memory_fit_observed'] is True
    assert proof['library_sha256'] == library_sha256 == sha(root / 'build/bin/garnet.dll')
    archive = root / 'results' / (name + '-evidence.zip')
    assert sha(archive) == proof['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for path, digest in manifest['files'].items():
            assert hashlib.sha256(z.read(path)).hexdigest() == digest
        closed = json.loads(z.read('results/' + name + '/workflow.json'))
    assert closed == json.loads((root / 'results' / name / 'workflow.json').read_bytes())
    assert type(closed['exit_code']) is int and closed['exit_code'] == 0
    assert closed['cleanup_finished'] is True and closed['library_unchanged'] is True
    assert closed['source_sha256'] == closed['source_after_sha256']
    assert closed['runtime_dependencies'] == closed['runtime_dependencies_after']
    candidate = 'Garnet/xModel/qwen35/vl_0_8b/cuda_mtp_prefill16_study.py'
    assert candidate in closed['source_sha256']
    for path, digest in closed['source_sha256'].items():
        assert sha(root / path) == digest, ('Qualified prefill16 source drift', path)
    for path, digest in closed['runtime_dependencies'].items():
        assert sha(root / 'build/bin' / path) == digest
    return {'full4k_archive_sha256': proof['archive_sha256'], 'full4k_proof_sha256': sha(proof_path),
            'candidate_source_sha256': closed['source_sha256'][candidate],
            'draft_prefill_chunk': 16, 'draft_decode_chunk': 4,
            'extra_draft_engine_weights_duplicated': True,
            'http_quality_or_performance_qualified': False}
