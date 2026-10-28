"""Physical-group host trace replay; never CUDA kernel or serving-speed evidence."""
import json,math,re
CORE={'worker_request','reset','session_prefill','session_generate','target_program_t4','target_program_t32','draft_decode_program','draft_prefill_program'}
def valid_id(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{32}',value) is not None
def admit(raw,expected_ids):
    assert type(expected_ids) in (set,frozenset) and expected_ids and all(valid_id(n) for n in expected_ids)
    rows=[];seen=set();spans_seen=set();active=None;previous_time=None
    for line in raw.splitlines():
        try:e=json.loads(json.loads(line)['line'])
        except (ValueError,TypeError,KeyError):continue
        if not isinstance(e,dict) or e.get('event') not in ('study_profile_group','study_stage'):continue
        assert e['CUDA_kernel_elapsed_claimed'] is False and e.get('clean_throughput_result_claimed',False) is False
        if e['event']=='study_profile_group':
            assert active is None and valid_id(e['id']);ids=e['request_ids'];flags=e['requested_profile_flags']
            assert type(ids) is list and 1<=len(ids)<=4 and e['id']==ids[0] and len(set(ids))==len(ids) and set(ids)<=set(expected_ids) and not seen.intersection(ids)
            assert type(e['real_rows']) is type(e['physical_batch']) is int and e['real_rows']==len(ids) and e['physical_batch']==4 and type(flags) is list and len(flags)==len(ids) and all(v is True for v in flags)
            assert e['measurement']=='host_wall_intervals_including_logging'
            active=dict(id=e['id'],request_ids=ids,sequence=0,stack=[],intervals=[],counts={},inclusive_seconds={});continue
        assert active is not None and e['id']==active['id'] and e['clock_domain']=='native_worker'
        assert e['measurement']=='host_wall_interval_including_instrumentation'
        assert type(e['sequence']) is int and e['sequence']==active['sequence']+1;active['sequence']=e['sequence']
        now=e['native_monotonic_seconds'];assert type(now) in (int,float) and math.isfinite(now) and (previous_time is None or now>=previous_time);previous_time=now
        label=e['stage'];span=e['span'];parent=e['parent_span'];assert isinstance(label,str) and re.fullmatch('[a-z][a-z0-9_]{0,63}',label) and type(span) is int and span>0 and (parent is None or type(parent) is int and parent>0)
        stack=active['stack']
        if e['phase']=='begin':
            assert span not in spans_seen and parent==(stack[-1]['span'] if stack else None);spans_seen.add(span)
            if not stack:assert label=='worker_request' and active['sequence']==1
            stack.append(dict(span=span,stage=label,parent_span=parent,started=now))
        else:
            assert e['phase']=='end' and stack and e['call_succeeded'] is True
            row=stack.pop();assert (row['span'],row['stage'],row['parent_span'])==(span,label,parent) and now>=row['started']
            row.update(finished=now,host_wall_seconds=now-row['started']);active['intervals'].append(row)
            active['counts'][label]=active['counts'].get(label,0)+1;active['inclusive_seconds'][label]=active['inclusive_seconds'].get(label,0)+row['host_wall_seconds']
            if not stack:
                assert label=='worker_request' and CORE<=set(active['counts']) and all(active['counts'][n]>0 for n in CORE)
                assert all(active['counts'][n]==1 for n in ('worker_request','reset','session_prefill','session_generate'))
                active.pop('stack');seen.update(active['request_ids']);rows.append(active);active=None
    assert active is None and seen==set(expected_ids) and rows
    return dict(status='B4_physical_group_host_stage_intervals_replayed',groups=rows,real_request_IDs=len(seen),CUDA_kernel_elapsed_or_clean_serving_speed_qualified=False,
                scope='Native-worker host wall only, including instrumentation. Each physical group maps to all real requests; stages are not per-row latency. Nested inclusive times cannot be added across labels. Async GPU work may be charged to later reads/waits. token_delivery counts include suppressed dummy/duplicate callbacks and are not public token counts. Absolute clocks from native, bridge and client cannot be subtracted.')
