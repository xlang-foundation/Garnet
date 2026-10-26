"""Replay extraction provenance and stage-derived vectors; weights stay external.

This checker does not claim local validation of absent checkpoint weight bytes.
"""
from pathlib import Path, PureWindowsPath
import argparse
import hashlib
import json
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_z_projection import vectors, N, K

JOB = 'gdn-z-fixture-extraction-attempt1'
PARENT = 'gdn-chunk-stage-witness-attempt2'
METHODS = 'Garnet/test2026/cuda/qwen38_5090/'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def checked(path):
    z = zipfile.ZipFile(path)
    manifest = json.loads(z.read('evidence-manifest.json'))
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
    for name, expected in manifest['files'].items():
        assert sha(z.read(name)) == expected
    return z, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    args = parser.parse_args()
    archive = args.proofs / (JOB + '-evidence.zip')
    z, manifest = checked(archive)
    with z:
        base = 'results/' + JOB + '/'
        workflow = json.loads(z.read(base + 'workflow.json'))
        assert workflow['status'] == 'real_gdn_z_128_fixture_extracted_native_control_unrun'
        assert workflow['exit_code'] == 0 and workflow['cleanup_finished'] is True
        assert workflow['before'] == workflow['after']
        assert workflow['gpu_or_native_model_executed'] is workflow['sdk_modified'] is workflow['model_or_performance_qualified'] is False
        for phase in ('cpu-ledger', 'extraction-ledger'):
            events = [json.loads(line) for line in z.read(base + phase + '/commands.jsonl').splitlines()]
            assert len(events) == 2 and events[0]['event'] == 'started' and events[1]['event'] == 'finished' and events[1]['exit_code'] == 0
            assert events[0]['source']['owned_source_sha256'] == events[1]['source_after']['owned_source_sha256']
            expected = {**workflow['before']['source'], **{PureWindowsPath(name).relative_to(PureWindowsPath('D:/CantorAI/5090-study-20261023')).as_posix(): h for name, h in workflow['before']['methods'].items()}}
            assert events[0]['source']['owned_source_sha256'] == expected
        assert b'Ran 6 tests' in z.read(base + 'cpu-ledger/commands.log') and b'\nOK' in z.read(base + 'cpu-ledger/commands.log')
        fixture_raw = z.read('results/actual-gdn-z-128-fixture/manifest.json')
        fixture = json.loads(fixture_raw)
        assert sha(fixture_raw) == workflow['fixture_manifest_sha256']
        assert fixture_raw == (args.proofs / (JOB + '-manifest.json')).read_bytes()
        proof_raw = z.read('results/' + PARENT + '-verified.json')
        assert proof_raw == (args.proofs / (PARENT + '-verified.json')).read_bytes()
        proof = json.loads(proof_raw)
        assert proof['status'] == 'gdn_chunk_stage_witness_arrays_replayed'
        assert fixture['parent_proof_sha256'] == workflow['parent_proof_sha256'] == sha(proof_raw)
        assert fixture['parent_archive_sha256'] == workflow['parent_archive_sha256'] == proof['archive_sha256'] == manifest['linked_parent_archives'][PARENT]
        assert sha((args.proofs / (PARENT + '-evidence.zip')).read_bytes()) == proof['archive_sha256']
        for name in ('gdn_z_projection.py', 'prepare_gdn_z_projection_fixture.py', 'test_gdn_z_projection.py'):
            assert z.read(METHODS + name) == Path(__file__).with_name(name).read_bytes()
        assert fixture['extractor_sha256'] == sha(z.read(METHODS + 'prepare_gdn_z_projection_fixture.py'))
        parent, unused = checked(args.proofs / (PARENT + '-evidence.zip'))
        with parent:
            source_base = 'results/' + PARENT + '/native/'
            report_raw = parent.read(source_base + 'report.json')
            assert sha(report_raw) == fixture['parent_report_sha256']
            report = json.loads(report_raw)
            prep = json.loads(parent.read(METHODS + 'evidence/gdn-chunk-stage-witness-bound-preparation.json'))
            assert fixture['config_sha256'] == prep['checkpoints']['models/Qwen3.8-27B-NVFP4/config.json']
            assert fixture['index_sha256'] == prep['checkpoints']['models/Qwen3.8-27B-NVFP4/model.safetensors.index.json']
            assert fixture['candidate_library_sha256'] == proof['candidate_library_sha256']
            assert len(fixture['layers']) == 2 and len(fixture['files']) == 12
            replayed = []
            for layer in (0, 1):
                values, comparison = vectors(report, lambda name: parent.read(source_base + name), layer)
                row = fixture['layers'][layer]
                assert all(row[key] == value for key, value in comparison.items())
                assert row['dimensions'] == {'m': [32, 128], 'n': N, 'k': K}
                for name, raw in values.items():
                    assert fixture['files']['layer' + str(layer) + '/' + name] == {'bytes': len(raw), 'sha256': sha(raw)}
                prefix = 'model.language_model.layers.' + str(layer) + '.linear_attn.in_proj_z.'
                assert set(row['tensors']) == {prefix + suffix for suffix in ('weight', 'weight_scale', 'input_scale')}
                for key, tensor in row['tensors'].items():
                    weight = key.endswith('.weight')
                    assert tensor['key'] == key and tensor['dtype'] == ('F8_E4M3' if weight else 'F32')
                    assert tensor['shape'] == ([N, K] if weight else [])
                    first, last = tensor['data_offsets']
                    size = N * K if weight else 4
                    assert 0 <= first < last and last - first == size
                    assert fixture['files']['layer' + str(layer) + '/' + tensor['file']] == {'bytes': size, 'sha256': tensor['sha256']}
                replayed.append(comparison)
    result = {'status': 'gdn_z_fixture_stage_vectors_and_extraction_provenance_replayed',
              'archive_sha256': sha(archive.read_bytes()), 'files_verified': len(manifest['files']),
              'fixture_manifest_sha256': sha(fixture_raw), 'parent_archive_sha256': proof['archive_sha256'],
              'stage_derived_vectors_verified': 6, 'comparisons': replayed,
              'remote_cpu_tests_passed': 6, 'source_runtime_sdk_and_checkpoint_metadata_unchanged': True,
              'checkpoint_weight_payload_verified_locally': False,
              'native_projection_control_executed': False,
              'model_or_quality_or_performance_qualified': False,
              'replay_verifier_sha256': sha(Path(__file__).read_bytes()),
              'scope': 'Complete stage-derived input/output vector hashes rebuilt from immutable parent; extraction ledger and tensor metadata checked. Actual weight/scale payload is external on study machine and has not been independently reread here. Requires actual native both-profile reproduction and full BF16/FP8 replay before a candidate algorithm, then unchanged full4K/quality/HTTP gates.'}
    target = args.proofs / (JOB + '-verified.json')
    raw = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    print(json.dumps({'status': result['status'], 'proof_sha256': sha(raw), 'vectors': 6}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
