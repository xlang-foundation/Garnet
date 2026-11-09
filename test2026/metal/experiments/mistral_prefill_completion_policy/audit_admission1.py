"""Audit candidate outputs and completion-first plans; timing is diagnostic."""
from pathlib import Path
import collections,copy,hashlib,json,math,runpy,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent;M=ROOT.parent;SG=M/'mistral-fused-mlp-serving1'
sys.path.insert(0,str(SG/'audit-tools1'))
from audit_half_transport1 import require,load_client,validate
from audit_policies1 import split_policy,profile32_policy,value_policy,paired_policy,silu_policy,timing
fused_check=runpy.run_path(str(SG/'audit1.py'))['check_fused']
files={}
def raw(p):
    data=p.read_bytes();files[str(p)]=hashlib.sha256(data).hexdigest();return data
def read(p):return json.loads(raw(p))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'admission-audit1.json'
require(not DEST.exists(),'never overwrite')
status=read(ROOT/'admission1-status.json');require(status['status']=='terminal' and status['returncode']==0,'unfinished collection')
require(read(ROOT/'admission1-postflight.json')['passed'],'missing postflight')
for name,h in read(ROOT/'admission1-preflight.json')['files_sha256'].items():require(sha(Path(name))==h,'changed admitted input '+name)
base=ROOT/'admission1';state=read(base/'status.json');phase=state['phases']['observed']
require(state['status']=='collected_pending_independent_audit' and phase['status']=='collected' and phase['shutdown']['graceful_native_closed'] and not phase['shutdown']['forced_kill'],'unclean engine')
folder=base/'observed';profile=read(folder/'profile.json');expected=read(ROOT/'profile1.json');expected.update(profile_id=expected['profile_id']+'-fresh',memory_inventory_path=str(folder/'prepared-inventory.json'))
require(profile==expected and profile['prefill_completion_priority'] is True and not profile['trace_stage_latency'],'wrong candidate profile')
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
    require(es.index(preparations[0])<es.index(ready_event) and 'native_prefill_completion_priority_optional' in ready_event['backend'],'wrong native policy backend/readiness')
    owner=None;ready=set();previous_owner=0;last_batch=None;owners=[];overlap=0;decode_width=0
    for e in es:
        kind=e['event']
        if kind=='native_batch':
            requests=e['requests'];ids={r['id'] for r in requests}
            if e['stage']=='decode':
                require(ids==ready and requests,'ready decoders were omitted or reordered behind prefill');decode_width=max(decode_width,len(requests))
            else:
                require(len(requests)==1 and e['prefill_chunk']==128,'completion-first plan has multiple/wide prefill owners')
                r=requests[0]
                if owner is None:
                    require(r['id']>previous_owner and r['prefill_offset']==0,'owner start bypasses FIFO/native identity order')
                    owner=r['id'];owners.append(owner)
                require(r['id']==owner and r['prefill_tokens']<=128,'owner changed before prompt completed')
                if ready:
                    require(last_batch is not None and last_batch['stage']=='decode' and {r['id'] for r in last_batch['requests']}==ready,'prefill ran before ready decode batch')
                    overlap+=1
            last_batch=e
        elif kind=='native_prefill_chunk' and e['final']:
            require(owner==e['native_request'],'final acknowledgment belongs to another owner')
            ready.add(owner);previous_owner=owner;owner=None
        elif kind=='completed':
            require(e['native_request'] in ready,'completed request not decode-ready');ready.remove(e['native_request'])
    require(owner is None and not ready and owners==list(range(1,13)) and overlap>0 and 2<=decode_width<=4,'unfinished policy or absent actual batching/overlap')
    return dict(prefill_owner_start_order=owners,prefill_steps_with_ready_decoders=overlap,observed_maximum_decode_batch_rows=decode_width,all_prefill_steps_one_owner=True,all_ready_decoders_scheduled_before_prefill=True)
policy=policy_check(events);observed=max(len(e['requests']) for e in events if e['event']=='native_batch')
def check(es,cs,ct):
    policy_check(es)
    report=validate(es,cs,ct,profile,exit_status,expected_requests=12,control_prefixes=False,maximum_chunk=128,prefix_policy=False,required_active_rows=observed)
    require(all(r['exact_native_coding_control'] for r in report['requests']),'missing full control equality')
    fused=fused_check(es,profile)
    for fn in [split_policy,profile32_policy,value_policy,paired_policy,silu_policy]:fn(es,profile)
    require(all(c['result']['generated_tokens']==128 and c['result']['cached_prompt_tokens']==0 for c in cs),'wrong output/cache lengths')
    return report,fused
native,fused=check(events,clients,controls);rejected=[]
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
for name,mut in [('changed_owner_mid_prompt',wrong_owner),('omitted_ready_decoder',drop_decode),('changed_output',wrong_output),('changed_control',wrong_control),('wrong_native_policy',wrong_policy),('stale_epoch',wrong_epoch),('false_ttft',false_ttft),('wrong_fusion',wrong_fusion)]:
    es,cs,ct=copy.deepcopy(events),copy.deepcopy(clients),copy.deepcopy(controls);mut(es,cs,ct)
    try:
        check(es,cs,ct)
        for r in ['cold','warm1','warm2']:timing([c for c in cs if c['round']==r])
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
result=dict(passed=True,scope=__doc__,requests=12,output_tokens=1536,all_ids_and_text_exact_frozen_fp32=True,policy=policy,native=native,fused=fused,diagnostic_metrics=metrics,negative_controls_rejected=rejected,files_sha256=files,
    actual_batch_width_reported=True,client_concurrency=4,canonical_gpu_kv_contents_independently_observed=False,prefix_cache_lineage_qualified=False,coding_quality_qualified=False,matched_control_campaign_collected=False,
    default_policy_enabled=False,production_runtime_integrated=False,serving_speed_claim=False,goal_complete=False)
DEST.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','native']},indent=2))
