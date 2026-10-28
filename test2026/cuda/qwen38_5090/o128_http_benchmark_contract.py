"""Raw fixed4K stream/memory replay and exact retained T32 IDs for all124 requests."""
import hashlib,json
from .compare_http_runs import verify_run
from .benchmark_fixed_length_native import final_ids,ARCHIVE_SHA256,MANIFEST_SHA256
from .cohort_memory_replay import replay_cohort_memory
sha=lambda b:hashlib.sha256(b).hexdigest()
REFERENCE='prefill16-fixed4k-http-attempt1'
REFERENCE_SHA='d64adb0cdba9d7c3446f15034b34d095025394815d28c4dae463b0056377e748'

def replay_trial(manifest,read,corpus,reference_manifest,reference_read):
    assert manifest['status']==reference_manifest['status']=='completed'
    assert manifest['corpus_archive_sha256']==reference_manifest['corpus_archive_sha256']==ARCHIVE_SHA256
    assert manifest['corpus_manifest_sha256']==reference_manifest['corpus_manifest_sha256']==MANIFEST_SHA256
    assert manifest['actual_input_tokens_required']==3968 and manifest['actual_completion_tokens_required']==128
    assert manifest['gpu']['uuid']==reference_manifest['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
    fresh=verify_run(manifest,read);assert len(fresh['cohorts'])==8
    raw=corpus.read('manifest.json');assert sha(raw)==MANIFEST_SHA256;rows=json.loads(raw)['rows'];assert len(rows)==124
    for row in rows:
        key=row['key'];request=corpus.read(row['request_file']);ids_raw=corpus.read(row['expected_prompt_ids_file']);assert sha(request)==row['request_sha256'] and sha(ids_raw)==row['expected_prompt_ids_sha256']
        ids=json.loads(ids_raw);prefix=manifest['run_id']+'-'+key;oldprefix=reference_manifest['run_id']+'-'+key
        assert read(prefix+'-request.json')==reference_read(oldprefix+'-request.json')==request
        events=[json.loads(line) for line in read(prefix+'-events.jsonl').splitlines()];old=[json.loads(line) for line in reference_read(oldprefix+'-events.jsonl').splitlines()]
        actual=final_ids(events,ids);control=final_ids(old,ids);assert actual['response_token_ids']==control['response_token_ids'],('O128 benchmark token difference',key)
    samples=[json.loads(line) for line in read(manifest['run_id']+'-vram.jsonl').splitlines()];memory={}
    for row in manifest['cohorts']:
        key=row['case']+'-c'+str(row['concurrency'])
        requests=[json.loads(read(name+'-result.json')) for name in row['request_keys']]
        memory[key]=replay_cohort_memory(samples,key,row,requests)
        assert row['prompt_tokens']==[3968]*8 and row['generated_tokens']==1024
    return dict(status='actual_O128_fixed4k_trial_raw_SSE_memory_and_all124_T32_IDs_replayed',requests=124,measured_requests=64,cohorts=fresh['cohorts'],memory=memory,performance_target_qualified=False)
