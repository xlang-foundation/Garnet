"""Replay host-stage diagnostics against retained cases; never a speed benchmark."""
import hashlib
import json
import re
import zipfile

from .replay_native_host_stage_trace import replay


def closed_observed_reference(root, name, digest):
    if not re.fullmatch(r'sglang-observed-fixed4k-ordinary-attempt[1-9][0-9]*', name):
        raise ValueError('Actual named observed-output reference required')
    if not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Actual closed archive SHA256 required')
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    archive = root / 'results' / (name + '-evidence.zip')
    assert sha(archive.read_bytes()) == digest
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('evidence-manifest.json'))
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist())) == len(manifest['files']) + 1
        for path, expected in manifest['files'].items():
            assert sha(z.read(path)) == expected
        raw = z.read('results/' + name + '/workflow.json')
    workflow = json.loads(raw)
    assert workflow == json.loads((root / 'results' / name / 'workflow.json').read_bytes())
    assert type(workflow['exit_code']) is int and type(workflow['server_exit_code']) is int
    assert workflow['cleanup_finished'] is True
    assert workflow['source_sha256'] == workflow['source_after_sha256']
    assert workflow['runtime_dependencies'] == workflow['runtime_dependencies_after']
    proof_path = root / 'results' / (name + '-verified.json')
    proof = json.loads(proof_path.read_bytes())
    status = ('reference_fixed4k_observed_profile_replayed' if workflow['exit_code'] == 0
              else 'reference_fixed4k_observed_failure_preserved')
    assert proof['archive_sha256'] == digest and proof['status'] == status
    return {'job': name, 'archive_sha256': digest, 'proof_sha256': sha(proof_path.read_bytes()),
            'proof_status': status, 'exit_code': workflow['exit_code'], 'cleanup_finished': True}


def replay_report(report, read, cases):
    assert report['status'] == 'native_host_stage_diagnostic_collected_needs_independent_replay'
    assert report['performance_admitted'] is False and set(report['modes']) == {'ordinary', 'mtp'}
    assert len(cases['cases']) == 4 and cases['orders'] == [[0, 1, 2, 3], [2, 0, 3, 1], [0, 1, 2, 3]]
    assert report['parent_archive_sha256'] == cases['parent_archive_sha256']
    results = {}
    for mode in ('ordinary', 'mtp'):
        record = report['modes'][mode]
        assert record['status'] == 'instrumented_host_stages_replayed'
        assert type(record['worker_exit_code']) is int and record['worker_exit_code'] == 0
        assert len(record['requests']) == 12
        expected = {}
        request_index = 0
        for repeat, order in enumerate(cases['orders']):
            for index in order:
                case = cases['cases'][index]
                key = mode + '-r' + str(repeat) + '-' + case['key']
                assert record['requests'][request_index] == {
                    'id': key, 'case_index': index, 'repeat': repeat,
                    'parent_request_key': case['key'], 'prompt_tokens': 3968,
                    'output_tokens': 128, 'retained_ids_exact': True}
                request_index += 1
                expected[key] = {'case': case['case'], 'prompt_ids': case['prompt_ids'],
                                 'generated_ids': case['generated_ids_by_mode'][mode],
                                 'image_sha256': case['image_sha256']}
        rows = [json.loads(line) for line in read(mode + '/host-stages.jsonl').splitlines()]
        events, ignored = [], []
        for line in read(mode + '/worker/worker-stdout.jsonl').splitlines():
            outer = json.loads(line)
            try:
                event = json.loads(outer['line'])
            except ValueError:
                ignored.append(outer)
                continue
            assert isinstance(event, dict) and event.get('event') in ('ready', 'token', 'completed')
            events.append(event)
        assert ignored == record['non_json_stdout_lines']
        assert events[0] == record['ready'] and events[0]['generation_mode'] == mode
        fresh = replay(rows, events, expected)
        assert fresh == record['replay']
        results[mode] = fresh
    return {'status': 'native_host_stage_report_replayed', 'modes': results,
            'requests': 24, 'performance_admitted': False,
            'isolated_gpu_or_kernel_time_measured': False, 'two_times_target_achieved': False}
