"""Replay every B8 MTP target state/vector/ID against actual closed ordinary bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.paged_batch8_mtp import PARENT, BASELINE, replay_mtp, validate_parent

JOB = 'paged-batch8-mtp-admission-attempt1'
METHODS = 'Garnet/test2026/cuda/qwen38_5090/'
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def checked_archive(path):
    z = zipfile.ZipFile(path)
    manifest = json.loads(z.read('evidence-manifest.json'))
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
    for name, digest in manifest['files'].items():
        assert sha(z.read(name)) == digest
    return z, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proofs', type=Path, required=True)
    args = parser.parse_args()
    archive = args.proofs / (JOB + '-evidence.zip')
    z, manifest = checked_archive(archive)
    old, _ = checked_archive(args.proofs / (PARENT + '-evidence.zip'))
    with z, old:
        doc = lambda name: json.loads(z.read(name))
        prefix = 'results/' + JOB + '/'
        raw = z.read(METHODS + 'evidence/paged-batch8-mtp-preparation.json')
        p = json.loads(raw)
        w = doc(prefix + 'workflow.json')
        assert sha(raw) == w['preparation_sha256'] and p['status'] == 'paged_batch8_mtp_prepared_gpu_unrun'
        assert w['source_sha256'] == w['source_after_sha256'] == p['files']
        assert w['runtime_dependencies'] == w['runtime_dependencies_after'] == p['runtime']
        assert w['environment_before'] == w['environment_after'] == p['environment']
        assert w['library_sha256'] == p['runtime']['garnet.dll'] == BASELINE
        assert w['cleanup_finished'] is w['library_unchanged'] is True and type(w['exit_code']) is int
        assert w['parents'] == p['parents'] and set(p['parents']) == {PARENT}
        for name, digest in p['new_files'].items():
            assert sha(z.read(name)) == digest
        for name in ('verify_paged_batch8_mtp.py', 'paged_batch8_mtp.py', 'paged_batch8_model.py',
                     'verify_bf16_batch8_admission.py', 'bf16_batch8_admission.py', 'bf16_packed_state_gate.py'):
            assert sha(Path(__file__).with_name(name).read_bytes()) == p['files'][METHODS + name]
        parent_raw = z.read('results/' + PARENT + '-verified.json')
        assert parent_raw == (args.proofs / (PARENT + '-verified.json')).read_bytes()
        q = json.loads(parent_raw)
        row = p['parents'][PARENT]
        parent_digest = sha((args.proofs / (PARENT + '-evidence.zip')).read_bytes())
        assert sha(parent_raw) == row['proof_sha256'] and q['status'] == row['status']
        assert parent_digest == row['archive_sha256'] == manifest['linked_parent_archives'][PARENT]
        closed = json.loads(old.read('results/' + PARENT + '/workflow.json'))
        validate_parent(q, closed, BASELINE, parent_digest)
        assert closed['runtime_dependencies_after'] == p['runtime'] and closed['environment_after'] == p['environment']
        assert p['files'] == {**closed['source_after_sha256'], **p['new_files']}
        passed = w['exit_code'] == 0
        replayed = memory = None
        if passed:
            assert w['status'] == 'paged_batch8_mtp_collected_needs_replay'
            for ledger in ('cpu-ledger', 'gpu-ledger'):
                rows = [json.loads(line) for line in z.read(prefix + ledger + '/commands.jsonl').splitlines()]
                assert len(rows) == 2 and rows[-1]['event'] == 'finished' and type(rows[-1]['exit_code']) is int and rows[-1]['exit_code'] == 0
            log = z.read(prefix + 'cpu-ledger/commands.log')
            assert ('Ran ' + str(p['expected_cpu_test_count']) + ' tests').encode() in log and b'\nOK' in log
            report = doc(prefix + 'model/report.json')
            reference = json.loads(old.read('results/' + PARENT + '/model/report.json'))
            assert report['ordinary_parent'] == {'job': PARENT, 'archive_sha256': parent_digest,
                'proof_sha256': sha(parent_raw), 'ordinary_model_preserved': True, 'mtp_or_http_or_performance_admitted': False}
            replayed = replay_mtp(report, lambda name: z.read(prefix + 'model/' + name), reference,
                                  lambda name: old.read('results/' + PARENT + '/model/' + name))
            assert replayed == report['retained_baseline_replay']
            resource = doc(prefix + 'resources/resource-monitor.json')
            samples = [json.loads(line) for line in z.read(prefix + 'resources/resource-vram.jsonl').splitlines()]
            assert type(resource['child_exit_code']) is int and resource['child_exit_code'] == 0 and resource['gpu']['uuid'] == p['gpu_uuid']
            assert samples and all(s['phase'] == 'construction-and-correctness' for s in samples)
            gaps = [b['monotonic'] - a['monotonic'] for a, b in zip(samples, samples[1:])]
            assert all(g > 0 for g in gaps)
            memory = resource['sampled_memory']
            assert memory['samples'] == len(samples) and memory['sampled_peak_adapter_used_bytes'] == max(s['used_bytes'] for s in samples)
            assert memory['observed_sample_gap_median_seconds'] == (statistics.median(gaps) if gaps else None)
            assert memory['observed_sample_gap_max_seconds'] == (max(gaps) if gaps else None)
        result = {'status': 'paged_batch8_mtp_target_state_replayed' if passed else 'paged_batch8_mtp_failure_preserved',
                  'archive_sha256': sha(archive.read_bytes()), 'files_verified': len(manifest['files']),
                  'library_sha256': BASELINE, 'ordinary_parent': row, 'mtp_target_state_replay': replayed,
                  'construction_correctness_adapter_sampled_memory': memory, 'external_b1_b4_math_qualified': False,
                  'broad_quality_qualified': False, 'http_qualified': False, 'serving_memory_saving_qualified': False,
                  'performance_admitted': False, 'two_times_target_achieved': False, 'scope': p['scope']}
    target = args.proofs / (JOB + '-verified.json')
    raw = json.dumps(result, indent=2).encode()
    assert not target.exists() or target.read_bytes() == raw
    if not target.exists():
        target.write_bytes(raw)
    print(json.dumps({'status': result['status'], 'archive_sha256': result['archive_sha256'],
                      'files_verified': result['files_verified'], 'proof_sha256': sha(raw), 'target_states': 29 if passed else None}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
