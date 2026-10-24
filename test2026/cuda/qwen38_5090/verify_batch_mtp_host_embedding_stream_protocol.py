"""Replay actual B2 worker transport against independently retained B1 oracles."""
from pathlib import Path
import base64
import hashlib
import json
import math,sys
import zipfile

out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / 'batch-mtp-host-embedding-stream-protocol-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read('evidence-manifest.json'))
    for name, digest in evidence['files'].items():
        assert sha(z.read(name)) == digest
    for name, digest in evidence['linked_archives'].items():
        assert sha((out / name).read_bytes()) == digest
    doc = lambda name: json.loads(z.read(name))
    base = 'results/batch-mtp-host-embedding-stream-protocol-attempt1/'
    workflow = doc(base+'workflow.json')
    report = doc(base+'native/report.json')
    assert workflow['exit_code'] == report['exit_code'] == 0
    assert report['status'] == 'batch_stream_protocol_passed_needs_independent_replay'
    build=doc('results/native-host-embedding-build-attempt1/manifest.json')
    assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
    assert evidence['library_sha256']==workflow['library_sha256']==build['library_sha256']
    for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
        for path,digest in build[key].items():assert sha(z.read('Garnet/'+path))==digest
    full_path='results/shared-mtp3-full-b2-host-embedding-host-attempt1-verified.json'
    full=doc(full_path)
    assert full['status']=='host_embedding_full_b2_host_regression_verified' and full['model_gate_passed'] and full['strict_target_state_hash_gate_passed'] and full['retained_old_library_state_exact'] and full['host_embedding_lookup']
    assert full['library_sha256']==evidence['library_sha256']
    assert full['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b2-host-embedding-host-attempt1-evidence.zip']
    assert workflow['host_embedding_lookup'] and workflow['full_host_proof_sha256']==sha(z.read(full_path))
    workspace=doc('results/shared-mtp3-full-b2-host-embedding-host-attempt1/workflow.json')
    assert workflow['workspace_source_sha256']==workspace['workspace_source_sha256']
    for path,digest in workspace['workspace_source_sha256'].items():assert sha(z.read('Garnet/'+path))==digest
    worker_path='Garnet/test2026/cuda/qwen38_5090/native_batch_mtp_host_embedding_stream_worker.py'
    probe_path='Garnet/test2026/cuda/qwen38_5090/probe_batch_mtp_host_embedding_stream_protocol.py'
    assert workflow['worker_sha256']==sha(z.read(worker_path)) and workflow['probe_sha256']==sha(z.read(probe_path))
    repair=doc('results/unicode-transport-repair2-verified.json')
    assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
    assert sha(z.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py'))==repair['repair_source_sha256']
    with zipfile.ZipFile(out/'batch-mtp-deferred-stream-protocol-attempt1-evidence.zip') as old:
        old_worker=old.read('Garnet/test2026/cuda/qwen38_5090/native_batch_mtp_deferred_stream_worker.py').decode('utf-8-sig').replace('\r\n','\n')
        new_worker=z.read(worker_path).decode('utf-8-sig').replace('\r\n','\n')
        assert old_worker[old_worker.index('    for frame in requests():'):]==new_worker[new_worker.index('    for frame in requests():'):]
        assert old.read('results/batch-mtp-deferred-stream-protocol-attempt1/native/requests.jsonl')==z.read(base+'native/requests.jsonl')

    wire = z.read(base+'native/requests.jsonl')
    assert sha(wire) == report['request_sha256']
    frames = [json.loads(line) for line in wire.splitlines()]
    names = ['s3-text-c1-r0-greedy', 's3-image-c1-r0-greedy', 'red', 'green']
    groups = [names[:2], names[:2], [names[0]], [names[1]], names[2:], names[:2]]
    assert len(frames) == 6
    expected = {}
    for index, (frame, group) in enumerate(zip(frames, groups)):
        assert len(frame['batch']) == len(group)
        for row, (job, name) in enumerate(zip(frame['batch'], group)):
            reference = 'results/native-mtp3-http/'+name
            body = doc(reference+'-request.json')
            choice = doc(reference+'-response.json')['choices'][0]
            if index == 5 and row == 0:
                body['max_tokens'] = 2
            key = str(index)+'-'+str(row)
            assert job == {'id': key, 'body': body}
            content = body['messages'][0]['content']
            image_hash = None if isinstance(content, str) else sha(base64.b64decode(content[0]['image_url']['url'].split(',', 1)[1], validate=True))
            expected[key] = {'prompt': choice['prompt_token_ids'], 'ids': choice['response_token_ids'][:body['max_tokens']], 'image_hash': image_hash}
    tokens = {key: [] for key in expected}
    completed = set()
    ready = []
    previous = -math.inf
    for line in z.read(base+'native/worker-stdout.log').decode('utf8').splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get('event')
        if kind == 'ready':
            ready.append(event)
            continue
        if kind not in ('token', 'completed', 'error'):
            continue
        assert len(ready) == 1 and kind != 'error'
        key = event['id']
        assert key in expected and key not in completed
        if kind == 'token':
            tokens[key].append(event['token_id'])
            now = event['native_monotonic_seconds']
            assert math.isfinite(now) and now >= previous
            previous = now
            assert event['sampled_tokens'] == len(tokens[key])
            assert event['prompt_tokens'] == len(expected[key]['prompt'])
        else:
            use_mtp = key.split('-')[0] in ('0', '1', '4')
            assert event['generation_mode'] == ('verified_mtp' if use_mtp else 'greedy_target')
            if use_mtp:
                counters = event['speculative_metrics']
                assert counters['target_forward_calls'] == counters['target_verification_calls'] + counters['target_prefix_recomputes']
                assert counters['target_verification_calls'] == len(counters['iterations'])
                assert 0 <= counters['accepted_draft_tokens'] <= counters['proposed_draft_tokens']
                assert counters['mtp_prefill_calls'] > 0
            assert event['generated_ids'] == tokens[key] == expected[key]['ids']
            assert event['prompt_ids'] == expected[key]['prompt']
            assert event['image_sha256'] == expected[key]['image_hash']
            completed.add(key)
    assert completed == set(expected) and len(completed) == 10
    assert ready[0] == report['ready'] and ready[0]['protocol'] == 'study_batch_v1' and ready[0]['batch'] == 2
    assert ready[0]['deferred_state_commit']
    assert ready[0]['host_embedding_lookup'] and ready[0]['embedding_mode']=='host_mapped_bf16' and ready[0]['embedding_table_bytes']==0
    assert ready[0]['mtp_depth'] == 3 and ready[0]['mtp_policy'] == 'two_real_rows_equal_limits_else_greedy'
    assert ready[0]['library_sha256'] == evidence['library_sha256'] and ready[0]['shared_compressed_weights']
    assert {(v['id'], v['tokens'], v['exact']) for v in report['cases']} == {(key, len(v['ids']), True) for key, v in expected.items()}
    events = [json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['exit_code'] == 0
    monitor = doc(base+'resources/resource-monitor.json')
    samples = [json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert monitor['child_exit_code'] == 0 and monitor['sampled_memory']['samples'] == len(samples)
    peak = max(v['used_bytes'] for v in samples)
    assert peak == monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
result = {'status': 'batch_mtp_host_embedding_stream_protocol_independently_verified', 'archive_sha256': sha(archive.read_bytes()),
    'files_verified': len(evidence['files']), 'library_sha256': evidence['library_sha256'],
    'host_embedding_lookup':True, 'scope':'Six groups/ten real B2 host-embedding MTP3 stream requests, singles/unequal-limit greedy fallback and source/full-model proof replay. Not HTTP, sanitizer, broad quality or performance.',
    'groups': 6, 'requests': 10, 'exact_ids_and_prompt_ids': True, 'partial_groups_and_row_isolation': True,
    'mtp_and_greedy_fallback_modes_verified': True, 'image_hashes_verified': True, 'sampled_peak_adapter_used_bytes': peak, 'performance_admitted': False}
(out/'batch-mtp-host-embedding-stream-protocol-attempt1-verified.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2))
