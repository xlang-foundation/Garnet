"""Compare full native row state across128-token and up-to512 schedules."""
from pathlib import Path
import collections,copy,hashlib,json,math,re,struct,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent;M=ROOT.parent;N=M/'mistral-fused-mlp-native1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def events(p):return [json.loads(s) for s in p.read_text().splitlines() if s.startswith('{')]
def select(es,kind):return [e for e in es if e['event']==kind]
def flatten(v):return [x for a in v for x in flatten(a)] if isinstance(v,list) else [v]
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'model-audit2.json'
need(not DEST.exists(),'never overwrite audit')
pre=read(ROOT/'model2-preflight.json');post=read(ROOT/'model2-postflight.json')
need(pre['passed'] and pre['performance_measurement'] is False and post['passed'] and post['all_engines_terminal'],'unfinished state collection')
for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'changed state input '+name)
statuses=[read(ROOT/('model-'+label+'2-status.json')) for label in ['control','wide']]
need(all(s['status']=='terminal' and s['returncode']==0 for s in statuses) and statuses[1]['started_ns']>=statuses[0]['finished_ns'] and statuses[0]['pid']!=statuses[1]['pid'],'not serial/fresh completed engines')
data={label:events(ROOT/('model-'+label+'2.stdout')) for label in ['control','wide']};frozen=events(N/'model1.stdout')
fixture=read(Path(read(ROOT/'profile-model-control2.json')['diagnostic_fixture']))
expected={e['id']:e for e in select(frozen,'completed')}
def verify(es,label):
    need(es[-1]['event']=='closed' and not select(es,'error'),'unclean model events')
    profile=read(ROOT/('profile-model-'+label+'2.json'));ready=select(es,'ready')
    need(len(ready)==1 and ready[0]['profile']==profile and 'half_attention_maximum_chunk=512' in ready[0]['backend'],'wrong model/backend profile')
    need(profile['prefill_completion_priority'] is (label=='wide') and profile['wide_owner_prefill'] is (label=='wide'),'wrong state comparison arm')
    need(select(es,'native_prefill_completion_policy')==[dict(event='native_prefill_completion_policy',enabled=(label=='wide'),owner_limit=1 if label=='wide' else 4,chunk=128)],'wrong configured native policy')
    done=select(es,'completed');need(len(done)==4 and {e['id'] for e in done}==set(expected),'wrong completed fixtures')
    outputs={e['id']:e for e in done};native={e['native_request']:e for e in done}
    for req in fixture['requests']:
        key='trained-'+str(req['row']);actual=outputs[key];saved=expected[key]
        for field in ['generated_ids','prompt_ids','text','cached_tokens','finish_reason']:need(actual[field]==saved[field],'frozen output differs '+key+' '+field)
        need(actual['prompt_ids']==req['token_ids'] and len(actual['generated_ids'])==8 and actual['cached_tokens']==0,'wrong prompt/output/cache bounds')
    batches=select(es,'native_batch');routes=select(es,'diagnostic_native_kv_hoist');half=select(es,'diagnostic_half_attention');fusion=select(es,'diagnostic_fused_mlp')
    need(len(batches)==len(routes)==len(half)==len(fusion)>0,'missing route records')
    for invocation,(b,r,h,f) in enumerate(zip(batches,routes,half,fusion),1):
        chunk=b['prefill_chunk'] if b['stage']=='prefill' else 1;ids=[0]*(4*chunk);active=[0]*(4*chunk);slots=[0.]*(4*chunk);epochs=[0]*4
        for q in b['requests']:
            out=native[q['id']];row=q['row'];epochs[row]=q['epoch'];need(0<=row<4 and q['epoch']>0,'invalid lease')
            count=q['prefill_tokens'] if b['stage']=='prefill' else 1
            for t in range(count):
                at=row*chunk+t;active[at]=1;slots[at]=float(q['prefill_offset']+t if b['stage']=='prefill' else q['context_length']-1)
                ids[at]=out['prompt_ids'][q['prefill_offset']+t] if b['stage']=='prefill' else out['generated_ids'][q['generated_tokens']-1]
        need(r['invocation']==h['invocation']==f['invocation']==invocation and r['ids']==ids and flatten(r['active'])==active and flatten(r['slots'])==slots and r['epochs']==epochs,'native tensors differ from frozen tokens/slots/leases')
        prefill=b['stage']=='prefill';need(r['dispatches']==(64 if prefill else 0) and r['mps_dispatches']==0,'incorrect commit hoist/MPS route')
        need(h['dispatches']==(32 if prefill else 0) and h['scratch_bytes']==68*1024*1024+384,'incorrect half route or workspace')
        blocks=32 if prefill and sum(active)>=128 else 0
        need(f==dict(event='diagnostic_fused_mlp',invocation=invocation,blocks=blocks,active_tokens=sum(active),logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks),'incorrect fused route')
        if label=='wide' and prefill:need(len(b['requests'])==1 and chunk in [2,4,8,16,32,64,128,256,512],'wrong prepared owner profile')
    witnesses=select(es,'diagnostic_request_state');need(len(witnesses)==8,'missing per-request states')
    keys={(e['id'],e['phase']) for e in witnesses};need(keys=={(key,phase) for key in outputs for phase in ['prefill','final']},'duplicate/missing state phases')
    for w in witnesses:
        out=outputs[w['id']];need((w['native_request'],w['row'],w['epoch'])==(out['native_request'],out['native_row'],out['native_epoch']),'state belongs to another lease')
        need(w['consumed_tokens']==len(out['prompt_ids'])+(7 if w['phase']=='final' else 0),'wrong witnessed prefix extent')
        need(len(w['kv'])==32 and all(len(pair)==2 and all(re.fullmatch('[0-9a-f]{64}',h) for h in pair) for pair in w['kv']),'incomplete canonical KV digests')
        values=flatten(w['hidden_values']);need(len(values)==4096 and all(math.isfinite(x) for x in values),'invalid complete hidden values')
        need(hashlib.sha256(struct.pack('<4096f',*values)).hexdigest()==w['hidden'],'hidden values/digest disagree')
    # Validate native leases at the exact point the snapshots were taken.
    leases={};last_epoch=[0]*4;progress={};states_seen=set()
    for e in es:
        if e['event']=='native_batch':
            for q in e['requests']:
                row=q['row'];identity=q['id']
                if row not in leases:
                    need(e['stage']=='prefill' and q['prefill_offset']==0 and q['epoch']>last_epoch[row],'row reuse without a fresh epoch')
                    leases[row]=(identity,q['epoch']);last_epoch[row]=q['epoch'];progress[identity]=0
                need(leases[row]==(identity,q['epoch']),'live physical owner changed')
                if e['stage']=='prefill':need(progress[identity]==q['prefill_offset'],'prefill progress gap');progress[identity]+=q['prefill_tokens']
                else:need(progress[identity]==q['context_length']-1,'decode progress gap');progress[identity]+=1
        elif e['event']=='diagnostic_request_state':
            need(leases.get(e['row'])==(e['native_request'],e['epoch']) and progress[e['native_request']]==e['consumed_tokens'],'snapshot outside live/consumed row prefix');states_seen.add((e['id'],e['phase']))
        elif e['event']=='completed':
            need((e['id'],'final') in states_seen and leases.get(e['native_row'])==(e['native_request'],e['native_epoch']),'completion before witnessed live state');del leases[e['native_row']]
    need(not leases,'live row after all completions')
    close=es[-1]['memory']
    need(all(close[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']) and 0<=close['metal_allocated_bytes']<=4194304,'postclose native/device owners retained')
    return dict(requests=4,tokens=32,per_request_states=8,per_request_digest_count=520,hidden_values=32768,fused_blocks=sum(e['blocks'] for e in fusion),half_dispatches=sum(e['dispatches'] for e in half),chunks=dict(collections.Counter(e['prefill_chunk'] for e in batches if e['stage']=='prefill')),postclose_metal_bytes=close['metal_allocated_bytes'])
def compare(c,w):
    result={label:verify(es,label) for label,es in [('control',c),('wide',w)]}
    # New control reproduces the frozen complete physical-state witnesses.
    need(select(c,'diagnostic_full_state')==select(frozen,'diagnostic_full_state'),'control changed frozen whole-engine state')
    for kind in ['native_batch','diagnostic_native_kv_hoist','diagnostic_half_attention','native_affine_profile','native_cooperative_affine','native_indexed','native_indexed_decode']:
        need(select(c,kind)==select(frozen,kind),'control changed frozen execution '+kind)
    by=lambda es:{(e['id'],e['phase']):e for e in select(es,'diagnostic_request_state')}
    ca,wa=by(c),by(w)
    for key in ca:
        for field in ['consumed_tokens','kv','hidden','hidden_values']:need(ca[key][field]==wa[key][field],'schedule changed full request state '+str(key)+' '+field)
    need(result['wide']['chunks'].get(512,0)>0 and result['wide']['chunks'].get(256,0)>0 and any(k<128 for k in result['wide']['chunks']),'wide and short-tail profiles not exercised')
    return result
result=compare(data['control'],data['wide']);rejected=[]
def wrong_kv(c,w):select(w,'diagnostic_request_state')[0]['kv'][0][0]='0'*64
def wrong_hidden(c,w):select(w,'diagnostic_request_state')[0]['hidden']='0'*64
def wrong_value(c,w):select(w,'diagnostic_request_state')[0]['hidden_values']=[[0.]*4096]
def wrong_epoch(c,w):select(w,'diagnostic_request_state')[0]['epoch']+=1
def wrong_extent(c,w):select(w,'diagnostic_request_state')[0]['consumed_tokens']-=1
def wrong_token(c,w):select(w,'completed')[0]['generated_ids'][0]^=1
def wrong_hoist(c,w):select(w,'diagnostic_native_kv_hoist')[0]['dispatches']=0
def missing_state(c,w):w.remove(select(w,'diagnostic_request_state')[0])
def wrong_control(c,w):select(c,'diagnostic_full_state')[0]['hidden']='0'*64
def owner_leak(c,w):w[-1]['memory']['garnet_live_matrix_owners']=1
for name,mutate in [('wrong_kv',wrong_kv),('wrong_hidden',wrong_hidden),('wrong_hidden_values',wrong_value),('stale_state_epoch',wrong_epoch),('wrong_consumed_extent',wrong_extent),('wrong_output',wrong_token),('missing_hoist',wrong_hoist),('missing_state',missing_state),('changed_frozen_control_state',wrong_control),('owner_leak',owner_leak)]:
    c,w=copy.deepcopy(data['control']),copy.deepcopy(data['wide']);mutate(c,w)
    try:compare(c,w)
    except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
    else:raise ValueError('corruption admitted '+name)
output=dict(passed=True,scope=__doc__,arms=result,all_output_tokens_text_exact_frozen=True,all_complete_request_kv_hidden_states_exact=True,canonical_kv_digest_words_covered=512*8*4096*128,per_request_state_digests_compared=520,per_request_hidden_values_compared=32768,frozen_whole_control_state_digests=260,frozen_whole_control_hidden_values=65536,
    negative_controls_rejected=rejected,preflight_sha256=sha(ROOT/'model2-preflight.json'),stdout_sha256={label:sha(ROOT/('model-'+label+'2.stdout')) for label in data},reference_stdout_sha256=sha(N/'model1.stdout'),
    prefix_cache_reuse_qualified=False,coding_quality_qualified=False,performance_measured=False,production_runtime_integrated=False,goal_complete=False)
DEST.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
