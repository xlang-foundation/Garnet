"""Real-worker EOF protocol/reset/row-isolation gate; no timing admission."""
import copy,base64,zipfile
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root, out = map(Path, sys.argv[1:3])
assert not out.exists()
out.mkdir()
oracle = root / 'results/native-mtp3-http'
keys = ['s3-text-c1-r0-greedy', 's3-image-c1-r0-greedy', 'red', 'green']
saved = {}
for key in keys:
    request = oracle / (key+'-request.json')
    response = oracle / (key+'-response.json')
    saved[key] = (json.loads(request.read_text()), json.loads(response.read_text())['choices'][0])
# Additional distinct real-image teachers come only from the independently
# verified C1 greedy corpus, not a speculative or concurrently batched output.
parent_name='batch-mtp-host-embedding-http-pilot-attempt1'
parent_proof=json.loads((root/'results'/(parent_name+'-verified.json')).read_text())
assert parent_proof['status']=='batch_mtp_host_embedding_http_pilot_independently_verified'
assert parent_proof['actual_dispatch_by_cohort']['image-c1']['greedy_requests']==8
assert parent_proof['actual_dispatch_by_cohort']['image-c1']['mtp_requests']==0
parent_archive=root/'results'/(parent_name+'-evidence.zip')
assert hashlib.sha256(parent_archive.read_bytes()).hexdigest()==parent_proof['archive_sha256']
teacher_sources={};image_hashes=[];id_hashes=[]
with zipfile.ZipFile(parent_archive) as archive:
    manifests=[n for n in archive.namelist() if '/benchmark/' in n and n.endswith('-manifest.json')];assert len(manifests)==1
    manifest=json.loads(archive.read(manifests[0]));prefix=manifests[0].rsplit('/',1)[0]+'/'
    cohort=next(c for c in manifest['cohorts'] if c['case']=='image' and c['concurrency']==1)
    image_names=[]
    for index,key in enumerate(cohort['request_keys'][:4]):
        request_path=prefix+key+'-request.json';events_path=prefix+key+'-events.jsonl'
        request_raw=archive.read(request_path);events_raw=archive.read(events_path);body=json.loads(request_raw)
        events=[json.loads(line)['data'] for line in events_raw.decode().splitlines()]
        assert events[-1]=='[DONE]'
        final=next(json.loads(value) for value in reversed(events) if value!='[DONE]')
        assert final['study']['generation_mode']=='greedy_target'
        assert len(final['response_token_ids'])==128 and len(final['prompt_token_ids'])==104
        image=base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1],validate=True)
        image_hashes.append(hashlib.sha256(image).hexdigest());id_hashes.append(hashlib.sha256(json.dumps(final['response_token_ids']).encode()).hexdigest())
        name='distinct-image-'+str(index);image_names.append(name)
        saved[name]=(body,{'prompt_token_ids':final['prompt_token_ids'],'response_token_ids':final['response_token_ids']})
        teacher_sources[name]={'request_path':request_path,'events_path':events_path,'request_sha256':hashlib.sha256(request_raw).hexdigest(),'events_sha256':hashlib.sha256(events_raw).hexdigest(),'image_sha256':image_hashes[-1],'generated_ids_sha256':id_hashes[-1]}
assert len(image_names)==len(set(image_hashes))==len(set(id_hashes))==4
(out/'teacher-provenance.json').write_text(json.dumps({'parent_archive_sha256':parent_proof['archive_sha256'],'teachers':teacher_sources},indent=2),encoding='utf8')
groups = [(keys, False), (keys, False), ([keys[0]], False),
    ([keys[1]], False), (keys[2:], False), (keys[:3], False), (keys[:2], True),
    (keys[:2]*2, False), (keys[:2]*2, False), (keys[2:]*2, False),
    (image_names,False), (image_names[:2]+[keys[0]],False), (image_names,True)]
frames = []
expected = {}
for index, (names, short_first) in enumerate(groups):
    jobs = []
    for row, name in enumerate(names):
        body = copy.deepcopy(saved[name][0])
        if short_first and row == 0:
            body['max_tokens'] = 2
        key = str(index)+'-'+str(row)
        jobs.append({'id': key, 'body': body})
        expected[key] = {'prompt': saved[name][1]['prompt_token_ids'],
            'generated': saved[name][1]['response_token_ids'][:body['max_tokens']]}
    use_mtp=len(jobs)==4 and len({job['body']['max_tokens'] for job in jobs})==1
    for job in jobs:expected[job['id']]['mode']='verified_mtp' if use_mtp else 'greedy_target'
    frames.append({'batch': jobs})
wire = ''.join(json.dumps(frame, separators=(',', ':'))+'\n' for frame in frames).encode()
(out / 'requests.jsonl').write_bytes(wire)
launcher = root / 'serve-native-kv-delta-worker.bat'
cache = json.loads((root / 'results/study-ssd-engine-cache/receipt.json').read_text())['cache_root']
process = subprocess.run(['cmd.exe', '/d', '/c', str(launcher), 'unused', cache], input=wire,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(out / 'worker-stdout.log').write_bytes(process.stdout)
(out / 'worker-stderr.log').write_bytes(process.stderr)
report = {'status': 'process_failed', 'exit_code': process.returncode, 'cases': [],
    'request_sha256': hashlib.sha256(wire).hexdigest(), 'performance_admitted': False}
def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
save()
assert process.returncode == 0, 'Worker failed; raw output retained'
tokens = {key: [] for key in expected}
completed = set()
ready = []
times = {key: [] for key in expected}
for line in process.stdout.decode('utf8', errors='strict').splitlines():
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
    key = event['id']
    assert key in expected and key not in completed, 'Unknown/dummy/retired row event'
    assert kind != 'error', event
    if kind == 'token':
        tokens[key].append(event['token_id'])
        times[key].append(event['native_monotonic_seconds'])
        assert event['sampled_tokens'] == len(tokens[key])
        assert event['prompt_tokens'] == len(expected[key]['prompt'])
    else:
        assert event['generation_mode'] == expected[key]['mode']
        assert event['generated_ids'] == tokens[key] == expected[key]['generated']
        assert event['prompt_ids'] == expected[key]['prompt']
        assert times[key] == sorted(times[key])
        completed.add(key)
        report['cases'].append({'id': key, 'exact': True, 'tokens': len(tokens[key])})
        save()
assert len(ready) == 1 and ready[0]['protocol'] == 'study_batch_v1' and ready[0]['batch'] == 4
assert ready[0]['mtp_depth']==3 and ready[0]['chunk']==ready[0]['prefill_chunk']==4 and ready[0]['host_embedding_lookup']
assert ready[0]['mtp_policy']=='four_real_rows_equal_limits_else_greedy' and ready[0]['embedding_table_bytes']==0
assert ready[0]['kv_delta_state'] and ready[0]['shared_output_storage']
assert completed == set(expected) and len(completed) == 40
report['status'] = 'batch_stream_protocol_passed_needs_independent_replay'
report['ready'] = ready[0]
save()
print(json.dumps(report, indent=2))
