"""Replay actual B4/T4 worker transport against independently retained B1 oracles."""
from pathlib import Path
import base64
import hashlib
import json
import math,sys
import zipfile

out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = out / 'batch4-t4-mtp-host-embedding-stream-protocol-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read('evidence-manifest.json'))
    assert len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for name, digest in evidence['files'].items():
        assert sha(z.read(name)) == digest
    for name, digest in evidence['linked_archives'].items():
        assert sha((out / name).read_bytes()) == digest
    doc = lambda name: json.loads(z.read(name))
    base = 'results/batch4-t4-mtp-host-embedding-stream-protocol-attempt1/'
    workflow = doc(base+'workflow.json')
    report = doc(base+'native/report.json')
    assert workflow['exit_code'] == report['exit_code'] == 0
    assert report['status'] == 'batch_stream_protocol_passed_needs_independent_replay'
    build=doc('results/native-host-embedding-build-attempt1/manifest.json')
    assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
    assert evidence['library_sha256']==workflow['library_sha256']==build['library_sha256']
    for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
        for path,digest in build[key].items():assert sha(z.read('Garnet/'+path))==digest
    full_path='results/shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1-verified.json'
    full=doc(full_path)
    assert full['status']=='host_embedding_mtp3_full_b4_t4_state_gate_verified' and full['model_gate_passed'] and full['strict_same_profile_target_state_hash_gate_passed'] and full['raw_last_hidden_arrays_verified']==8 and full['draft_builder_workspace_bytes']==256<<20
    assert full['library_sha256']==evidence['library_sha256']
    assert full['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1-evidence.zip']
    assert workflow['host_embedding_lookup'] and workflow['full_host_proof_sha256']==sha(z.read(full_path))
    workspace=doc('results/shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1/workflow.json')
    assert workflow['workspace_source_sha256']==workspace['workspace_source_sha256']
    for path,digest in workspace['workspace_source_sha256'].items():assert sha(z.read('Garnet/'+path))==digest
    worker_path='Garnet/test2026/cuda/qwen38_5090/native_batch4_t4_mtp_host_embedding_stream_worker.py'
    probe_path='Garnet/test2026/cuda/qwen38_5090/probe_batch4_t4_mtp_host_embedding_stream_protocol.py'
    assert workflow['worker_sha256']==sha(z.read(worker_path)) and workflow['probe_sha256']==sha(z.read(probe_path))
    repair=doc('results/unicode-transport-repair2-verified.json')
    assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
    assert sha(z.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py'))==repair['repair_source_sha256']
    wire = z.read(base+'native/requests.jsonl')
    assert sha(wire) == report['request_sha256']
    prep=doc('Garnet/test2026/cuda/qwen38_5090/evidence/b4-t4-host-mtp-stream-preparation.json')
    assert sha(wire)==prep['request_wire_sha256'] and prep['groups']==13 and prep['requests']==40
    with zipfile.ZipFile(out/'batch4-t4-stream-protocol-attempt1-evidence.zip') as old:
        old_wire=old.read('results/batch4-t4-stream-protocol-attempt1/native/requests.jsonl')
        assert wire.startswith(old_wire)
    names = ['s3-text-c1-r0-greedy', 's3-image-c1-r0-greedy', 'red', 'green']
    saved={name:(doc('results/native-mtp3-http/'+name+'-request.json'),doc('results/native-mtp3-http/'+name+'-response.json')['choices'][0]) for name in names}
    provenance=doc(base+'native/teacher-provenance.json')
    parent=doc('results/batch-mtp-host-embedding-http-pilot-attempt1-verified.json')
    assert parent['status']=='batch_mtp_host_embedding_http_pilot_independently_verified'
    assert parent['actual_dispatch_by_cohort']['image-c1']['mtp_requests']==0 and parent['actual_dispatch_by_cohort']['image-c1']['greedy_requests']==8
    assert provenance['parent_archive_sha256']==parent['archive_sha256']==evidence['linked_archives']['batch-mtp-host-embedding-http-pilot-attempt1-evidence.zip']
    images=[];image_hashes=[];id_hashes=[]
    with zipfile.ZipFile(out/'batch-mtp-host-embedding-http-pilot-attempt1-evidence.zip') as source:
        for index in range(4):
            name='distinct-image-'+str(index);teacher=provenance['teachers'][name]
            request=source.read(teacher['request_path']);events=source.read(teacher['events_path'])
            assert sha(request)==teacher['request_sha256'] and sha(events)==teacher['events_sha256']
            body=json.loads(request);values=[json.loads(line)['data'] for line in events.decode().splitlines()]
            assert values[-1]=='[DONE]'
            final=next(json.loads(value) for value in reversed(values) if value!='[DONE]')
            assert final['study']['generation_mode']=='greedy_target' and len(final['response_token_ids'])==128 and len(final['prompt_token_ids'])==104
            image=base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1],validate=True)
            assert sha(image)==teacher['image_sha256'] and sha(json.dumps(final['response_token_ids']).encode())==teacher['generated_ids_sha256']
            images.append(name);image_hashes.append(sha(image));id_hashes.append(teacher['generated_ids_sha256'])
            saved[name]=(body,{'prompt_token_ids':final['prompt_token_ids'],'response_token_ids':final['response_token_ids']})
    assert len(set(image_hashes))==len(set(id_hashes))==4
    groups=[(names,False),(names,False),([names[0]],False),([names[1]],False),(names[2:],False),(names[:3],False),(names[:2],True),(names[:2]*2,False),(names[:2]*2,False),(names[2:]*2,False),(images,False),(images[:2]+[names[0]],False),(images,True)]
    frames=[json.loads(line) for line in wire.splitlines()];assert len(frames)==13
    expected={}
    for index,(frame,(group,short)) in enumerate(zip(frames,groups)):
        assert len(frame['batch'])==len(group)
        for row,(job,name) in enumerate(zip(frame['batch'],group)):
            body=json.loads(json.dumps(saved[name][0]));choice=saved[name][1]
            if short and row==0:body['max_tokens']=2
            key=str(index)+'-'+str(row);assert job=={'id':key,'body':body}
            content=body['messages'][0]['content']
            image_hash=None if isinstance(content,str) else sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True))
            expected[key]={'prompt':choice['prompt_token_ids'],'ids':choice['response_token_ids'][:body['max_tokens']],'image_hash':image_hash}
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
            use_mtp = key.split('-')[0] in ('7', '8', '9', '10')
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
    assert completed == set(expected) and len(completed) == 40
    assert ready[0] == report['ready'] and ready[0]['protocol'] == 'study_batch_v1' and ready[0]['batch'] == 4
    assert ready[0]['deferred_state_commit']
    assert ready[0]['host_embedding_lookup'] and ready[0]['embedding_mode']=='host_mapped_bf16' and ready[0]['embedding_table_bytes']==0
    assert ready[0]['mtp_depth'] == 3 and ready[0]['mtp_policy'] == 'four_real_rows_equal_limits_else_greedy'
    assert ready[0]['library_sha256'] == evidence['library_sha256'] and ready[0]['shared_compressed_weights']
    assert {(v['id'], v['tokens'], v['exact']) for v in report['cases']} == {(key, len(v['ids']), True) for key, v in expected.items()}
    events = [json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[-1]['event'] == 'finished' and events[-1]['exit_code'] == 0
    monitor = doc(base+'resources/resource-monitor.json')
    samples = [json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert monitor['child_exit_code'] == 0 and monitor['sampled_memory']['samples'] == len(samples)
    peak = max(v['used_bytes'] for v in samples)
    assert peak == monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
result = {'status': 'batch4_t4_mtp_host_embedding_stream_protocol_independently_verified', 'archive_sha256': sha(archive.read_bytes()),
    'files_verified': len(evidence['files']), 'library_sha256': evidence['library_sha256'],
    'host_embedding_lookup':True, 'scope':'13 groups/40 real B4/T4 host-embedding MTP3 stream requests, singles/unequal-limit greedy fallback and source/full-model proof replay. Not HTTP, sanitizer, broad quality or performance.',
    'groups': 13, 'requests': 40, 'exact_ids_and_prompt_ids': True, 'partial_groups_and_row_isolation': True,
    'mtp_and_greedy_fallback_modes_verified': True, 'image_hashes_verified': True, 'sampled_peak_adapter_used_bytes': peak, 'performance_admitted': False}
(out/'batch4-t4-mtp-host-embedding-stream-protocol-attempt1-verified.json').write_text(json.dumps(result, indent=2), encoding='utf8')
print(json.dumps(result, indent=2))
