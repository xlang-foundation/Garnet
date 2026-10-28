"""Independent public-wire replay against real-row B1 prompt/output IDs."""
import json,math
from .fixed_length_policy import generation_stop_ids


def replay_public(raw,jobs,prompt_ids,expected_ids,stop_ids,image_hashes,metrics):
    assert type(raw) is bytes and raw.endswith(b'\n') and 1<=len(jobs)<=4
    n=len(jobs);keys=[j['id'] for j in jobs];assert len(set(keys))==n and len(prompt_ids)==len(expected_ids)==len(image_hashes)==n
    by_key={key:i for i,key in enumerate(keys)};tokens=[[] for _ in jobs];texts=['']*n;done=[False]*n;last_time=[None]*n;events=0
    for line in raw.splitlines():
        event=json.loads(line);assert type(event) is dict and event['id'] in by_key,'Unexpected public/dummy request ID'
        row=by_key[event['id']];assert not done[row];kind=event['event'];policy=generation_stop_ids(jobs[row]['body'],stop_ids);events+=1
        if kind=='token':
            token=event['token_id'];assert type(token) is int and 0<=token<248320
            at=len(tokens[row]);assert at<len(expected_ids[row]) and token==expected_ids[row][at]
            assert type(event['sampled_tokens']) is int and event['sampled_tokens']==at+1
            assert type(event['prompt_tokens']) is int and event['prompt_tokens']==len(prompt_ids[row])
            assert type(event['stop']) is bool and event['stop']==(token in policy) and isinstance(event['text'],str)
            clock=event['native_monotonic_seconds'];assert type(clock) in (int,float) and math.isfinite(clock) and clock>=0
            assert last_time[row] is None or clock>=last_time[row];last_time[row]=clock
            tokens[row].append(token);texts[row]=event['text']
        else:
            assert kind=='completed' and tokens[row]==expected_ids[row]==event['generated_ids']
            assert all(type(t) is int for t in event['generated_ids']+event['prompt_ids'])
            assert event['prompt_ids']==prompt_ids[row] and event['text']==texts[row]
            assert all(type(event[k]) is int for k in ('physical_batch','real_rows','padded_rows'))
            assert event['physical_batch']==4 and event['real_rows']==n and event['padded_rows']==4-n
            assert event['padding_policy']=='duplicate_first_real_row_no_public_dummy_events'
            assert event['generation_mode']=='verified_mtp' and event['speculative_metrics']==metrics
            assert type(event['ignore_eos']) is bool and event['ignore_eos']==jobs[row]['body'].get('ignore_eos',False)
            assert event['finish_reason']==('stop' if expected_ids[row][-1] in policy else 'length')
            assert not any(t in policy for t in expected_ids[row][:-1])
            assert event['image_sha256']==image_hashes[row] and event['image_grid']==([1,16,16] if image_hashes[row] else None)
            elapsed=event['native_elapsed_seconds'];assert type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0
            done[row]=True
    assert all(done) and tokens==expected_ids
    return dict(real_rows=n,padded_rows=4-n,public_ID_values=sum(map(len,tokens)),public_events=events,
                all_public_prompt_and_output_IDs_exact=True,no_extra_public_request_IDs=True,
                HTTP_or_latency_or_throughput_qualified=False)
