"""Replay native callback IDs and dispatch frames without trusting summary counts."""
import base64,hashlib,json,math,re

def events(raw,ready):
    tokens={};outputs={};announced=[]
    for line in raw.splitlines():
        try:value=json.loads(json.loads(line)['line'])
        except (ValueError,TypeError,KeyError):continue
        if not isinstance(value,dict):continue
        kind=value.get('event')
        if kind=='ready':announced.append(value)
        elif kind=='token':
            key=value['id'];assert re.fullmatch('[0-9a-f]{32}',key) and key not in outputs
            assert type(value['token_id']) is int and 0<=value['token_id']<248320
            tokens.setdefault(key,[]).append(value['token_id']);assert value['sampled_tokens']==len(tokens[key])
            assert math.isfinite(value['native_monotonic_seconds'])
        elif kind=='completed':
            key=value['id'];assert key not in outputs and value['generated_ids']==tokens[key]
            assert value['physical_batch']==4 and type(value['real_rows']) is int and 1<=value['real_rows']<=4
            assert value['padded_rows']==4-value['real_rows'] and value['generation_mode']=='verified_mtp'
            assert value['padding_policy']=='duplicate_first_real_row_no_public_dummy_events'
            assert math.isfinite(value['native_elapsed_seconds']) and value['native_elapsed_seconds']>=0
            outputs[key]=value
        elif kind=='error':raise AssertionError(value)
    assert announced==[ready] and set(tokens)==set(outputs)
    return outputs

def match_native(event,body,prompt,output,finish):
    assert event['prompt_ids']==prompt and event['generated_ids']==output and event['finish_reason']==finish
    assert event['ignore_eos'] is body.get('ignore_eos',False)
    assert output and len(output)<=body['max_tokens'] and len(prompt)+len(output)<=4096
    if body.get('ignore_eos',False):assert len(output)==body['max_tokens'] and finish=='length'
    content=body['messages'][0]['content']
    if isinstance(content,str):assert event['image_sha256'] is None and event['image_grid'] is None
    else:
        data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)
        assert event['image_sha256']==hashlib.sha256(data).hexdigest() and event['image_grid']==[1,16,16]
    metrics=event['speculative_metrics'];assert isinstance(metrics,dict) and isinstance(metrics['iterations'],list)
    if len(output)>1:assert metrics['iterations']

def dispatches(raw,bodies,outputs):
    seen=set();sizes=[];real=[]
    for line in raw.splitlines():
        row=json.loads(line);keys=row['request_ids'];assert row['physical_capacity']==4 and row['real_rows']==len(keys) and 1<=len(keys)<=4
        assert len(set(keys))==len(keys) and not seen.intersection(keys) and set(keys)<=set(bodies)
        assert row['grouping_clock']=='perf_counter'
        a,b,c=(row[n] for n in ('grouping_started_perf_counter','grouping_deadline_perf_counter','dispatch_perf_counter'))
        assert all(math.isfinite(n) for n in (a,b,c)) and abs((b-a)-.010)<1e-5 and c>=a
        for key in keys:
            assert row['request_created_perf_counter'][key]<=row['queue_put_started_perf_counter'][key]<=c
        # A dispatcher frame may be partitioned by output-limit/stop policy.
        groups={}
        for key in keys:groups.setdefault((bodies[key]['max_tokens'],bodies[key].get('ignore_eos',False)),[]).append(key)
        for group in groups.values():
            for key in group:assert outputs[key]['real_rows']==len(group)
        wire=(json.dumps({'batch':[dict(id=key,body=bodies[key]) for key in keys]},separators=(',',':'))+'\n').encode()
        sizes.append(len(wire));real.append(len(keys));seen.update(keys)
    assert seen==set(bodies)==set(outputs)
    return dict(dispatches=len(sizes),real_rows=real,max_joined_frame_bytes=max(sizes),joined_frames_over_default_2MiB=sum(n>2*1024*1024 for n in sizes))
