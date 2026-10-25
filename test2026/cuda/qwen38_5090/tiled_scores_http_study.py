"""Explicit candidate/baseline identities and restoration for the tiled HTTP trial."""
import hashlib
import json
import zipfile
from .native_tiled_scores_http_admission import BASELINE, CANDIDATE, MODEL, MODEL_ARCHIVE, validate_model

sha = lambda raw: hashlib.sha256(raw).hexdigest()


def validate_lifecycle(w, p):
    assert type(w['exit_code']) is int and w['cleanup_finished'] is True
    assert w['source_sha256'] == w['source_after_sha256'] == p['files']
    assert w['runtime_baseline'] == w['restored_runtime'] == p['runtime_baseline']
    assert w['runtime_baseline']['garnet.dll'] == BASELINE
    assert w['runtime_dependencies'] == p['runtime_dependencies'] == {**w['runtime_baseline'], 'garnet.dll': CANDIDATE}
    assert w['library_sha256'] == CANDIDATE and w['baseline_library_restored'] is True
    assert w['environment_before'] == w['environment_after'] == p['environment']
    assert w['selector_before'] == w['selector_after'] and w['performance_admitted'] is False
    assert type(w['candidate_library_installed']) is bool
    expected = w['runtime_dependencies'] if w['candidate_library_installed'] else w['runtime_baseline']
    assert w['runtime_after_before_restore'] == expected
    assert w['tiled_model_parent'] == p['tiled_model_parent']
    if w['exit_code'] == 0:
        assert w['candidate_library_installed'] is True
        assert w['admission']['candidate_library_sha256'] == CANDIDATE
        assert w['admission']['baseline_library_sha256'] == BASELINE
        assert w['admission']['tiled_scores_enabled'] is True
        assert w['admission']['tiled_model_archive_sha256'] == MODEL_ARCHIVE
        assert w['admission']['tiled_model_proof_sha256'] == p['tiled_model_parent']['proof_sha256']


def replay_tiled_parent(proofs, current_archive, w, p):
    """Bind the actual closed model proof; never rewrite its old DLL identity."""
    row = p['tiled_model_parent']
    raw = current_archive.read('results/' + MODEL + '-verified.json')
    assert raw == (proofs / (MODEL + '-verified.json')).read_bytes()
    q = json.loads(raw)
    archive = proofs / (MODEL + '-evidence.zip')
    assert sha(raw) == row['proof_sha256']
    assert sha(archive.read_bytes()) == row['archive_sha256'] == MODEL_ARCHIVE
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for name, digest in manifest['files'].items():
            assert sha(z.read(name)) == digest
        closed = json.loads(z.read('results/' + MODEL + '/workflow.json'))
    validate_model(q, closed, CANDIDATE)
    assert closed['runtime_dependencies_after'] == p['runtime_baseline']
    assert closed['environment_after'] == p['environment']
    for name, digest in closed['source_after_sha256'].items():
        assert p['files'][name] == digest, ('Admitted model source drift', name)
    if w['exit_code'] == 0:
        assert w['admission']['tiled_model_proof_sha256'] == sha(raw)
    return q
