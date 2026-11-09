"""Audit exact outputs and native up-to512 owner profiles; timing is diagnostic."""
from pathlib import Path
import collections,copy,hashlib,json,math,runpy,statistics,sys
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;SG=M/'mistral-fused-mlp-serving1'
sys.path.insert(0,str(ROOT/'audit-tools1'))
from audit_half_transport1 import require,load_client,validate
from audit_policies1 import split_policy,profile32_policy,value_policy,paired_policy,silu_policy,timing
def decode_check(events,profile):
    require(profile['fused_decode_rows4'] is True,'missing native decode policy')
    ready=[e for e in events if e['event']=='ready']
    require(len(ready)==1 and 'native_fused_q4_decode_rows4_default_v1' in ready[0]['backend'],'wrong native decode backend')
    prep=[e for e in events if e['event']=='native_fused_q4_decode_prepared']
    require(len(prep)==1 and prep[0]['sites']==[1]*32 and type(prep[0]['profiles']) is int and 1<=prep[0]['profiles']<=2 and prep[0]['scratch_bytes']==65536 and prep[0]['extra_weight_bytes']==0 and events.index(prep[0])<events.index(ready[0]),'wrong native decode preparation')
    pending=None;frames=full=partial=blocks=copies=0
    for event in events:
        if event['event']=='native_batch':
            require(pending is None,'missing native decode observation')
            pending=event
        elif event['event']=='native_fused_q4_decode':
            require(pending is not None and event['stage']==pending['stage'],'unbound native decode observation')
            actual=len(pending['requests']) if pending['stage']=='decode' else 0
            wanted=32 if actual==4 else 0
            require(type(event['blocks']) is int and event['blocks']==wanted and type(event['input_copies']) is int and 0<=event['input_copies']<=wanted,'wrong native decode route/copies')
            frames+=1;blocks+=wanted;copies+=event['input_copies']
            full+=actual==4;partial+=pending['stage']=='decode' and actual<4;pending=None
    require(pending is None and full>0 and partial>0 and blocks==32*full,'missing full/partial native decode coverage')
    return dict(frames=frames,actual_four_request_decode_frames=full,partial_decode_frames=partial,native_fused_blocks=blocks,native_input_copies=copies,prepared_profiles=prep[0]['profiles'],private_scratch_bytes=65536,extra_weight_bytes=0)

def fused_check(events,profile):
    require(profile['fused_mlp_prefill'] is True,'missing fused policy')
    ready=[e for e in events if e['event']=='ready'];require(len(ready)==1 and 'fused_mistral_mlp_prefill_fast_v1' in ready[0]['backend'],'wrong fused backend')
    prep=[e for e in events if e['event']=='native_fused_mlp_prepared']
    require(prep==[dict(event='native_fused_mlp_prepared',sites=[1]*32,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1')] and events.index(prep[0])<events.index(ready[0]),'wrong fused preparation')
    batches=[e for e in events if e['event']=='native_batch'];fused=[e for e in events if e['event']=='native_fused_mlp'];spans=[e for e in events if e['event']=='native_compact_prefill']
    require(len(batches)==len(fused)==len(spans)>0,'missing fused/compact observations')
    maximum_parent=maximum_gpu=0
    for batch,event,span in zip(batches,fused,spans):
        prefill=batch['stage']=='prefill';chunk=batch['prefill_chunk'] if prefill else 1
        parent=4*chunk;active=sum(r['prefill_tokens'] if prefill else 1 for r in batch['requests'])
        rows=sorted(r['row'] for r in batch['requests'] if not prefill or r['prefill_tokens'])
        compact=prefill and len(rows) in [1,2] and len(rows)*chunk>4 and (len(rows)==1 or rows[1]==rows[0]+1)
        require(span==dict(event='native_compact_prefill',stage=batch['stage'],rows=len(rows) if compact else 0,first=rows[0] if compact else 4,chunk=chunk),'wrong compact owner mapping')
        gpu=(len(rows) if compact else 4)*chunk
        require(0<active<=gpu<=512 and parent>=gpu,'effective GPU projection exceeds compiled token workspace')
        blocks=32 if prefill and active>=128 and gpu>4 else 0
        # This worker field reports the full B4 input parent, before compaction.
        require(event==dict(event='native_fused_mlp',blocks=blocks,active_tokens=active,physical_rows=parent,logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks),'wrong fused route/parent metadata')
        maximum_parent=max(maximum_parent,parent);maximum_gpu=max(maximum_gpu,gpu)
    require(sum(e['blocks'] for e in fused)>0,'fusion never executed')
    return dict(blocks=sum(e['blocks'] for e in fused),producer_kernels=sum(e['producer_kernels'] for e in fused),down_kernels=sum(e['down_kernels'] for e in fused),maximum_declared_input_parent_rows=maximum_parent,maximum_effective_gpu_rows=maximum_gpu,parent_metadata_is_not_gpu_dispatch_extent=True)

files={}
def raw(p):
    data=p.read_bytes();files[str(p)]=hashlib.sha256(data).hexdigest();return data
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
require(not (ROOT/'admission-audit1.json').exists(),'never overwrite')
status=read(ROOT/'admission1-status.json');require(status['status']=='terminal' and status['returncode']==0,'unfinished collection')
require(read(ROOT/'admission1-postflight.json')['passed'],'missing postflight')
for name,h in read(ROOT/'admission1-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed admitted input '+name)
base=ROOT/'admission1';state=read(base/'status.json');phase=state['phases']['observed']
require(state['status']=='collected_pending_independent_audit' and phase['status']=='collected' and phase['shutdown']['graceful_native_closed'] and not phase['shutdown']['forced_kill'],'unclean engine')
folder=base/'observed';profile=read(folder/'profile.json');expected=read(ROOT/'profile-admission1.json');expected.update(profile_id=expected['profile_id']+'-fresh',memory_inventory_path=str(folder/'prepared-inventory.json'))
require(profile==expected and profile['prefill_completion_priority'] is True and profile['wide_owner_prefill'] is True and profile['maximum_owner_prefill_chunk']==512 and not profile['trace_stage_latency'],'wrong candidate profile')
manifest=read(base/'manifest.json');serving=read(Path(manifest['source_manifest']));plan=next(e for e in serving['plans'] if e['id']=='p2048-c4-o128')
bodies=read(Path(plan['cases']));workload=read(Path(plan['workload']));prompts=[e['token_ids'] for e in workload['requests']]
controls_raw=[json.loads(json.loads(line)['line']) for line in raw(Path(manifest['engine']['saved_native_control'])).splitlines()]
controls={}
for i,prompt in enumerate(prompts):
    matches=[e for e in controls_raw if e['event']=='completed' and e['prompt_ids']==prompt and len(e['generated_ids'])==128]
    require(matches and all(e['generated_ids']==matches[0]['generated_ids'] and e['text']==matches[0]['text'] for e in matches),'missing/inconsistent frozen control')
    controls['p2048-c4-o128-'+str(i)]=dict(status='completed',prompt_ids=prompt,generated_ids=matches[0]['generated_ids'],text=matches[0]['text'])
clients=[]
for i,entry in enumerate(manifest['requests']):
    row=i%4;require(entry['body']==bodies[row] and entry['case_id']=='p2048-c4-o128-'+str(row),'changed request')
    prefix=folder/entry['label']/'request-0';c=load_client(prefix,entry['case_id'],entry['body'],prompts[row],read,raw);c['round']=entry['round']
    c['wire']=[json.loads(line) for line in raw(prefix.with_suffix('.events.jsonl')).splitlines()];clients.append(c)
require(len(clients)==12 and [c['round'] for c in clients]==[r for r in ['cold','warm1','warm2'] for _ in range(4)],'wrong round coverage')
events=[json.loads(json.loads(line)['line']) for line in raw(folder/'server/worker-stdout.jsonl').splitlines()]
exit_status=read(folder/'server/worker-exit.json')
def policy_check(es):
    preparations=[e for e in es if e['event']=='native_prefill_completion_policy']
    require(preparations==[dict(event='native_prefill_completion_policy',enabled=True,owner_limit=1,chunk=128)],'wrong/missing native policy preparation')
    ready_event=next(e for e in es if e['event']=='ready')
    require(es.index(preparations[0])<es.index(ready_event) and 'native_prefill_completion_priority_optional' in ready_event['backend'] and 'half_attention_maximum_chunk=512' in ready_event['backend'],'wrong native policy backend/readiness')
    owner=None;ready=set();previous_owner=0;last_batch=None;owners=[];overlap=0;decode_width=0;chunks=collections.Counter();ticks=0;maximum_tick_tokens=0
    for e in es:
        kind=e['event']
        if kind=='native_profile_tick':
            ticks+=1
            total=sum(len(b['requests']) if b['stage']=='decode' else sum(r['prefill_tokens'] for r in b['requests']) for b in e['batches'])
            require(0<total<=512,'native tick exceeds actual global token budget')
            maximum_tick_tokens=max(maximum_tick_tokens,total)
        elif kind=='native_batch':
            requests=e['requests'];ids={r['id'] for r in requests}
            if e['stage']=='decode':
                require(ids==ready and requests,'ready decoders were omitted or reordered behind prefill');decode_width=max(decode_width,len(requests))
            else:
                require(len(requests)==1 and e['prefill_chunk'] in [2,4,8,16,32,64,128,256,512],'completion-first plan has multiple/unprepared prefill owners')
                r=requests[0];remaining=r['prompt_tokens']-r['prefill_offset'];budget=512-len(ready)
                wanted=2
                if remaining<=128:
                    while wanted<remaining and wanted<512:wanted*=2
                else:
                    while wanted*2<=min(remaining,512):wanted*=2
                chosen=wanted
                while chosen>=2 and min(remaining,chosen)>budget:chosen//=2
                require(chosen>=2 and e['prefill_chunk']==chosen and r['prefill_tokens']==min(remaining,chosen),'native shape differs from prepared up-to profile/budget choice')
                chunks[e['prefill_chunk']]+=1
                if owner is None:
                    require(r['id']>previous_owner and r['prefill_offset']==0,'owner start bypasses FIFO/native identity order')
                    owner=r['id'];owners.append(owner)
                require(r['id']==owner and 0<r['prefill_tokens']<=e['prefill_chunk'],'owner changed before prompt completed')
                if ready:
                    require(last_batch is not None and last_batch['stage']=='decode' and {r['id'] for r in last_batch['requests']}==ready,'prefill ran before ready decode batch')
                    overlap+=1
            last_batch=e
        elif kind=='native_prefill_chunk' and e['final']:
            require(owner==e['native_request'],'final acknowledgment belongs to another owner')
            ready.add(owner);previous_owner=owner;owner=None
        elif kind=='completed':
            require(e['native_request'] in ready,'completed request not decode-ready');ready.remove(e['native_request'])
    require(owner is None and not ready and owners==list(range(1,13)) and overlap>0 and 2<=decode_width<=4 and chunks[512]>0 and chunks[256]>0 and any(c<128 for c in chunks) and ticks>0,'unfinished policy or missing wide/tail/batch coverage')
    return dict(prefill_owner_start_order=owners,prefill_steps_with_ready_decoders=overlap,observed_maximum_decode_batch_rows=decode_width,all_prefill_steps_one_owner=True,all_ready_decoders_scheduled_before_prefill=True,prepared_owner_chunk_histogram=dict(chunks),profile_ticks=ticks,maximum_actual_tick_tokens=maximum_tick_tokens)
policy=policy_check(events);observed=max(len(e['requests']) for e in events if e['event']=='native_batch')
def check(es,cs,ct):
    policy_check(es)
    decode_check(es,profile)
    report=validate(es,cs,ct,profile,exit_status,expected_requests=12,control_prefixes=False,maximum_chunk=128,prefix_policy=False,required_active_rows=observed)
    require(all(r['exact_native_coding_control'] for r in report['requests']),'missing full control equality')
    fused=fused_check(es,profile)
    for fn in [split_policy,profile32_policy,value_policy,paired_policy,silu_policy]:fn(es,profile)
    require(all(c['result']['generated_tokens']==128 and c['result']['cached_prompt_tokens']==0 for c in cs),'wrong output/cache lengths')
    return report,fused
native,fused=check(events,clients,controls);decode=decode_check(events,profile);rejected=[]
def wrong_owner(es,cs,ct):
    q=[e for e in es if e['event']=='native_batch' and e['stage']=='prefill'];q[1]['requests'][0]['id']+=1
def drop_decode(es,cs,ct):
    next(e for e in es if e['event']=='native_batch' and e['stage']=='decode')['requests'].pop()
def wrong_output(es,cs,ct):next(e for e in es if e['event']=='completed')['generated_ids'][0]^=1
def wrong_control(es,cs,ct):next(iter(ct.values()))['generated_ids'][0]^=1
def wrong_policy(es,cs,ct):next(e for e in es if e['event']=='native_prefill_completion_policy')['enabled']=False
def wrong_epoch(es,cs,ct):next(e for e in es if e['event']=='native_batch')['requests'][0]['epoch']+=1
def false_ttft(es,cs,ct):cs[0]['result']['ttft_content_seconds']=0
def wrong_fusion(es,cs,ct):next(e for e in es if e['event']=='native_fused_mlp')['blocks']+=1
for name,mut in [('changed_owner_mid_prompt',wrong_owner),('omitted_ready_decoder',drop_decode),('changed_output',wrong_output),('changed_control',wrong_control),('wrong_native_policy',wrong_policy),('stale_epoch',wrong_epoch),('false_ttft',false_ttft),('wrong_fusion',wrong_fusion),('wrong_fused_parent',lambda es,cs,ct:next(e for e in es if e['event']=='native_fused_mlp').update(physical_rows=512)),('wrong_compact_extent',lambda es,cs,ct:next(e for e in es if e['event']=='native_compact_prefill').update(rows=4)),('unprepared_chunk',lambda es,cs,ct:next(e for e in es if e['event']=='native_batch' and e['stage']=='prefill').update(prefill_chunk=513)),('tick_budget_overflow',lambda es,cs,ct:next(b for b in next(e for e in es if e['event']=='native_profile_tick')['batches'] if b['stage']=='prefill')['requests'][0].update(prefill_tokens=513))]:
    es,cs,ct=copy.deepcopy(events),copy.deepcopy(clients),copy.deepcopy(controls);mut(es,cs,ct)
    try:
        check(es,cs,ct)
        for r in ['cold','warm1','warm2']:timing([c for c in cs if c['round']==r])
    except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
    else:raise ValueError('corruption admitted: '+name)
def delete_decode_observation(es,cs,ct):
    es.remove(next(e for e in es if e['event']=='native_fused_q4_decode'))
def missing_decode_preparation(es,cs,ct):
    es.remove(next(e for e in es if e['event']=='native_fused_q4_decode_prepared'))
for name,mut in [
    ('missing_decode_observation',delete_decode_observation),
    ('missing_decode_preparation',missing_decode_preparation),
    ('missing_four_row_fusion',lambda es,cs,ct:next(e for e in es if e['event']=='native_fused_q4_decode' and e['blocks']==32).update(blocks=0)),
    ('fused_partial_or_prefill',lambda es,cs,ct:next(e for e in es if e['event']=='native_fused_q4_decode' and e['blocks']==0).update(blocks=32)),
    ('false_decode_input_copy',lambda es,cs,ct:next(e for e in es if e['event']=='native_fused_q4_decode').update(input_copies=33)),
    ('unprepared_decode_sites',lambda es,cs,ct:next(e for e in es if e['event']=='native_fused_q4_decode_prepared')['sites'].__setitem__(0,0)),
]:
    es,cs,ct=copy.deepcopy(events),copy.deepcopy(clients),copy.deepcopy(controls);mut(es,cs,ct)
    try:check(es,cs,ct)
    except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
    else:raise ValueError('corruption admitted: '+name)
metrics={}
for r in ['cold','warm1','warm2']:
    cs=[c for c in clients if c['round']==r];row=timing(cs)
    row['ranked_ttft_seconds']=sorted(c['result']['ttft_content_seconds'] for c in cs)
    row['worst_request_ttft_seconds']=max(row['ranked_ttft_seconds'])
    gaps=[]
    for c in cs:
        previous=None;count=0
        for w in c['wire']:
            if w['data']=='[DONE]':continue
            e=json.loads(w['data']);u=e.get('usage')
            if u and u['completion_tokens']>count:
                if previous is not None:gaps.append(w['monotonic']-previous)
                count=u['completion_tokens'];previous=w['monotonic']
    require(len(gaps)==4*127 and all(math.isfinite(x) and x>=0 for x in gaps),'wrong observed token-gap coverage')
    row['observed_token_delivery_gap_p95_seconds']=sorted(gaps)[math.ceil(.95*len(gaps))-1]
    row['observed_token_delivery_gap_max_seconds']=max(gaps);metrics[r]=row
result=dict(passed=True,scope=__doc__,requests=12,output_tokens=1536,all_ids_and_text_exact_frozen_fp32=True,policy=policy,native=native,fused=fused,decode=decode,diagnostic_metrics=metrics,negative_controls_rejected=rejected,files_sha256=files,
    actual_batch_width_reported=True,client_concurrency=4,prepared_owner_maximum_chunk=512,actual_global_token_budget=512,canonical_gpu_kv_contents_independently_observed=True,native_model_state_gate_sha256=sha(ROOT/'model-audit2.json'),prefix_cache_lineage_qualified=False,coding_quality_qualified=False,matched_control_campaign_collected=False,
    default_policy_enabled=False,production_runtime_integrated=False,serving_speed_claim=False,goal_complete=False)
(ROOT/'admission-audit1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','native']},indent=2))
