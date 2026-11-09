"""Audit long-prefix HTTP reuse, native leases, complete row state and live survivors.

Untimed correctness only. Concurrent seed submission order is not presumed: all
three seed prefill acknowledgments must precede every newcomer native plan.
"""
from pathlib import Path
import collections,copy,hashlib,json,math,re,struct,sys
HERE=Path(__file__).resolve().parent
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'lifecycle-audit3.json'
J=ROOT.parent/'mistral-prefill-completion-wide1';SG=ROOT.parent/'mistral-fused-mlp-serving1'
sys.path.insert(0,str(HERE/'audit-tools1'))
from audit_half_transport1 import require,validate,load_client,OWNERS
from audit_fused_route1 import fused_check
from audit_policies1 import split_policy,profile32_policy,value_policy,paired_policy,silu_policy
files={}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def raw(p):
    v=p.read_bytes();files[str(p)]=hashlib.sha256(v).hexdigest();return v
def read(p):return json.loads(raw(p))
def events(path):return [json.loads(json.loads(line)['line']) for line in raw(path).splitlines()]
def select(es,kind):return [e for e in es if e['event']==kind]
def bylabel(d):return {c['plan']:next(e for e in d['events'] if e['event']=='completed' and e['id']==c['id']) for c in d['clients']}
def check_policy(es):
    require(select(es,'native_prefill_completion_policy')==[dict(event='native_prefill_completion_policy',enabled=True,owner_limit=1,chunk=128)],'wrong native completion policy')
    ready_event=select(es,'ready')[0]
    require('half_attention_maximum_chunk=512' in ready_event['backend'] and 'native_prefill_completion_priority_optional' in ready_event['backend'],'wrong wide/policy backend')
    ready=set();owner=None;previous_owner=0;last=None;chunks=collections.Counter();overlap=0;max_decode=0;ticks=0
    for e in es:
        kind=e['event']
        if kind=='native_profile_tick':
            tokens=sum(len(b['requests']) if b['stage']=='decode' else sum(q['prefill_tokens'] for q in b['requests']) for b in e['batches'])
            require(0<=tokens<=512 and tokens==e['token_count'],'tick token bound/packing differs');ticks+=1
        elif kind=='native_batch':
            if e['stage']=='decode':
                require({q['id'] for q in e['requests']}==ready and e['requests'],'ready decoder omitted');max_decode=max(max_decode,len(e['requests']))
            else:
                require(len(e['requests'])==1,'multiple prefill owners');q=e['requests'][0]
                if owner is None:
                    require(q['id']>previous_owner and q['prefill_offset']==q['cached_tokens'],'FIFO/cache owner start differs');owner=q['id']
                require(q['id']==owner,'owner changed mid-prompt')
                if ready:
                    require(last is not None and last['stage']=='decode' and {q['id'] for q in last['requests']}==ready,'prefill preceded ready decoders');overlap+=1
                chunks[e['prefill_chunk']]+=1
            last=e
        elif kind=='native_prefill_chunk' and e['final']:
            require(owner==e['native_request'],'acknowledged different prefill owner');ready.add(owner);previous_owner=owner;owner=None
        elif kind=='completed':
            require(e['native_request'] in ready,'completion before prefill ack');ready.remove(e['native_request'])
    require(owner is None and not ready and ticks>0 and max_decode==4 and chunks[512]>0 and chunks[256]>0 and overlap>0,'missing wide/batched/overlap coverage')
    return dict(ticks=ticks,maximum_decode_rows=max_decode,prefill_steps_with_ready_decoders=overlap,chunks=dict(chunks))
def check_arm(d,fixture,controls,state):
    es=d['events'];cached=d['label']=='cached';profile=d['profile'];done=bylabel(d)
    require(state['status']=='collected' and state['shutdown']['graceful_native_closed'] is True and state['shutdown']['forced_kill'] is False,'unclean HTTP engine')
    transport=validate(es,d['clients'],controls,profile,d['exit'],expected_requests=9,maximum_chunk=128,prefix_policy=cached,required_active_rows=4)
    policy=check_policy(es);fused=fused_check(es,profile)
    for fn in [split_policy,profile32_policy,value_policy,paired_policy,silu_policy]:fn(es,profile)
    expected=fixture['expected_cached_tokens'];states={};first={};completed_at={};prefill_at={}
    witnesses=select(es,'native_request_state_witness');require(len(witnesses)==18,'missing/extra target state witness')
    for entry in fixture['requests']:
        name=entry['label'];c=done[name];want=expected.get(name,0) if cached else 0
        require(c['cached_tokens']==want,'wrong retained prefix length '+name)
        plans=[(i,e,q) for i,e in enumerate(es) if e['event']=='native_batch' for q in e['requests'] if q['id']==c['native_request']]
        require(plans and plans[0][1]['stage']=='prefill' and plans[0][2]['prefill_offset']==want,'wrong restore start '+name)
        first[name]=plans[0][0];completed_at[name]=next(i for i,e in enumerate(es) if e['event']=='completed' and e['id']==c['id'])
        prefill_at[name]=next(i for i,e in enumerate(es) if e['event']=='native_prefill_chunk' and e['native_request']==c['native_request'] and e['final'])
        if cached and name=='A-return':require(plans[0][2]['prefill_tokens']==0,'full hit executed prompt body')
        rows=[e for e in witnesses if e['id']==c['id']];require([e['phase'] for e in rows]==['prefill','final'],'wrong target phase order')
        states[name]={}
        for row in rows:
            require((row['native_request'],row['row'],row['epoch'])==(c['native_request'],c['native_row'],c['native_epoch']),'target state lease differs')
            st=row['state'];require(len(st['kv'])==32 and all(len(v)==2 for v in st['kv']),'incomplete target KV')
            require(all(re.fullmatch('[0-9a-f]{64}',v) for pair in st['kv'] for v in pair),'bad target digest')
            values=st['hidden_values'];require(len(values)==4096 and all(type(x) in (float,int) and math.isfinite(x) for x in values),'invalid target hidden values')
            require(hashlib.sha256(struct.pack('<4096f',*values)).hexdigest()==st['hidden'],'hidden values/hash mismatch')
            consumed=len(c['prompt_ids'])+(len(c['generated_ids'])-1 if row['phase']=='final' else 0)
            require(st['consumed_tokens']==consumed,'target consumed offset differs')
            states[name][row['phase']]=st
    seeds=['B','C','D'];chain=fixture['newcomer_chain'];seedids={str(done[n]['native_request']) for n in seeds}
    # Actual native restore/execution is the critical boundary; arrival order is
    # intentionally observed, never inferred from the client barrier or label D.
    require(max(prefill_at[n] for n in seeds)<first['A'],'A executed before all three seed prefills finished')
    require(len({done[n]['native_row'] for n in seeds+['A']})==4,'initial live owners overlap')
    admitE=[i for i,e in enumerate(es) if e['event']=='native_stage_trace' and e.get('stage')=='admit' and e.get('id')==done['E']['id']]
    require(len(admitE)==1 and first['A']<admitE[0]<completed_at['A']<first['E'],'fifth E did not queue behind four live owners')
    survivor_witnesses=select(es,'native_survivor_restore_witness');require(survivor_witnesses,'missing live survivor witnesses')
    observations=0;covered=set()
    for at,e in enumerate(es):
        if e['event']!='native_survivor_restore_witness':continue
        require(e['before']==e['after'] and e['before'],'new lease/restore changed live state')
        for identity,val in e['before'].items():
            c=next((c for c in done.values() if str(c['native_request'])==identity),None)
            require(c is not None and (val['row'],val['epoch'])==(c['native_row'],c['native_epoch']),'survivor identity/epoch mismatch')
            require(len(val['kv'])==32 and all(len(pair)==2 and all(re.fullmatch('[0-9a-f]{64}',v) for v in pair) for pair in val['kv']) and re.fullmatch('[0-9a-f]{64}',val['hidden']),'incomplete survivor state')
            observations+=2*65
            if cached:
                require(type(val['tokens']) is int and val['tokens']>=0 and type(val['prefill_tokens']) is int and 0<=val['prefill_tokens']<=len(c['prompt_ids']),'invalid survivor tracked offsets')
        for name in chain:
            if done[name]['native_request'] in e['pending_requests'] and set(e['before'])==seedids and max(prefill_at[n] for n in seeds)<at<first[name]:
                require(at+1<len(es) and es[at+1]['event']=='native_profile_tick' and any(q['id']==done[name]['native_request'] for batch in es[at+1]['batches'] if batch['stage']=='prefill' for q in batch['requests']),'survivor observation not bound to newcomer restore tick')
                covered.add(name)
                if cached:
                    require(all(e['before'][str(done[n]['native_request'])]['prefill_tokens']==len(done[n]['prompt_ids']) and e['before'][str(done[n]['native_request'])]['tokens']>=len(done[n]['prompt_ids']) for n in seeds),'restore witness has partially-prefilled survivor')
    require(covered==set(chain),'missing three-survivor restore witness for a newcomer')
    for name in chain:
        require(first[name]<min(completed_at[n] for n in seeds),'newcomer began after a survivor finished')
        require(completed_at[name]<min(completed_at[n] for n in seeds),'newcomer did not finish with all survivors live')
        require(done[name]['native_row']==done['A']['native_row'],'newcomer did not reuse A row')
    require(all(done[a]['native_epoch']<done[b]['native_epoch'] for a,b in zip(chain,chain[1:])),'A row reused stale epoch')
    require(states['A']['prefill']==states['A-return']['prefill'],'exact A return changed full prefill state')
    stats=select(es,'native_prefix_stats')
    if cached:
        require(stats and stats[-1]['stats']['hits']==3 and stats[-1]['stats']['misses']==6 and stats[-1]['stats']['prefill_capture_skipped_mixed']==9,'wrong native hit/miss/capture counts')
        require(stats[-1]['stats']['dense_capture_reused_blocks']>0 and stats[-1]['stats']['shared_resource_count']>0,'shared/direct cache never exercised')
        require(transport['head_only_prefill_batches']==1,'full exact hit did not use head-only plan')
    else:require(not stats and transport['head_only_prefill_batches']==0,'uncached arm retained state')
    return dict(transport=transport,policy=policy,fused=fused,cached_tokens={n:c['cached_tokens'] for n,c in done.items()},survivor_witnesses=len(survivor_witnesses),survivor_digest_observations=observations,covered_three_live_survivors=sorted(covered),cache=stats[-1]['stats'] if stats else None),states
def verify(data,fixture,controls,status):
    reports={};states={}
    for label in ['uncached','cached']:reports[label],states[label]=check_arm(data[label],fixture,controls,status[label])
    require(states['uncached']==states['cached'],'cached/uncached complete KV/hidden states differ')
    return reports
def main():
    dest=DEST;require(not dest.exists(),'never overwrite audit')
    require(sha(ROOT/'audit_lifecycle2.py')=='1789ada03cb6030cfdab9eb6ae047e7517a520b6ce57eac7defeef31e4016300','changed frozen lifecycle auditor')
    for p in sorted((HERE/'audit-tools1').glob('*.py')):
        require(p.read_bytes().rstrip()+b'\n'==(ROOT/'audit-tools1'/p.name).read_bytes().rstrip()+b'\n','changed replay helper '+p.name)
    require(read(ROOT/'launch1-status.json')['returncode']==0 and read(ROOT/'postflight1.json')['passed'],'collection unfinished')
    for name,h in read(ROOT/'preflight-hashes1.json')['files_sha256'].items():require(sha(Path(name))==h,'changed bound input '+name)
    for p in sorted((ROOT/'audit-tools1').glob('*.py'))+[ROOT/'prepare_audit1.py',ROOT/'audit-preparation1.json',ROOT/'audit_lifecycle1.py',ROOT/'audit_lifecycle2.py',ROOT/'collection-clarification1.json']:raw(p)
    fixture=read(ROOT/'fixture1.json');base=read(ROOT/'profile-template1.json');status=read(ROOT/'status1.json')
    require(status['status']=='collected_pending_independent_audit' and [s['label'] for s in status['phases']]==['uncached','cached'],'wrong campaign phases')
    data={}
    for label in ['uncached','cached']:
        folder=ROOT/label;profile=read(folder/'profile.json');expected=copy.deepcopy(base);expected.update(prefix_cache=label=='cached',native_cached_chunks=label=='cached',profile_id=base['profile_id']+'-'+label,memory_inventory_path=str(folder/'engine-inventory1.json'))
        require(profile==expected,'changed phase profile');clients=[load_client(folder/e['label']/'request-0',e['label'],e['body'],e['prompt_ids'],read,raw) for e in fixture['requests']]
        data[label]=dict(label=label,profile=profile,clients=clients,events=events(folder/'server/worker-stdout.jsonl'),exit=read(folder/'server/worker-exit.json'))
    uncached=bylabel(data['uncached']);controls={n:dict(status='completed',prompt_ids=c['prompt_ids'],generated_ids=c['generated_ids'],text=c['text']) for n,c in uncached.items()}
    # An independent frozen FP32 control anchors all unchanged long prompts.
    manifest=read(SG/'serving-manifest1.json');saved=events(Path(read(J/'admission2/manifest.json')['engine']['saved_native_control']))
    for n in ['B','C','D','A','A-return']:
        c=uncached[n];matches=[e for e in saved if e['event']=='completed' and e['prompt_ids']==c['prompt_ids']]
        require(matches and all(e['generated_ids'][:min(128,len(c['generated_ids']))]==c['generated_ids'][:128] for e in matches),'frozen native control prefix differs '+n)
    phases={s['label']:s for s in status['phases']};reports=verify(data,fixture,controls,phases)
    rejected=[]
    def event(d,kind):return next(e for e in d['cached']['events'] if e['event']==kind)
    def wrong_survivor(d):
        e=event(d,'native_survivor_restore_witness');next(iter(e['after'].values()))['hidden']='0'*64
    def wrong_return_plan(d):
        identity=bylabel(d['cached'])['A-return']['native_request']
        next(q for e in d['cached']['events'] if e['event']=='native_batch' for q in e['requests'] if q['id']==identity)['prefill_tokens']=1
    def wrong_return_offset(d):
        identity=bylabel(d['cached'])['A-return']['native_request']
        next(q for e in d['cached']['events'] if e['event']=='native_batch' for q in e['requests'] if q['id']==identity)['prefill_offset']-=1
    def missing_newcomer_witness(d):
        identity=bylabel(d['cached'])['changed-head']['native_request']
        d['cached']['events']=[e for e in d['cached']['events'] if not(e['event']=='native_survivor_restore_witness' and identity in e['pending_requests'])]
    mutations={
        'changed_output':lambda d:event(d,'completed')['generated_ids'].__setitem__(0,-1),
        'false_cache_miss':lambda d:next(c for c in d['cached']['clients'] if c['plan']=='A-return')['result'].update(cached_prompt_tokens=0),
        'false_early_edit_hit':lambda d:next(c for c in d['cached']['clients'] if c['plan']=='changed-head')['result'].update(cached_prompt_tokens=256),
        'changed_prompt':lambda d:next(c for c in d['cached']['clients'] if c['plan']=='A-return')['expected_prompt_ids'].__setitem__(0,-1),
        'wrong_full_hit_body':wrong_return_plan,'wrong_restored_offset':wrong_return_offset,
        'changed_survivor':wrong_survivor,'missing_newcomer_survivor_witness':missing_newcomer_witness,
        'stale_target_epoch':lambda d:event(d,'native_request_state_witness').update(epoch=0),
        'wrong_target_kv':lambda d:event(d,'native_request_state_witness')['state']['kv'][0].__setitem__(0,'0'*64),
        'wrong_hidden_values':lambda d:event(d,'native_request_state_witness')['state']['hidden_values'].__setitem__(0,1e10),
        'wrong_consumed_offset':lambda d:event(d,'native_request_state_witness')['state'].update(consumed_tokens=0),
        'overbudget_cache':lambda d:event(d,'native_prefix_stats')['stats'].update(accounted_host_bytes=2684354561),
        'owner_leak':lambda d:event(d,'closed')['memory'].update(garnet_live_matrix_owners=1),
        'wrong_gpu_extent':lambda d:event(d,'native_compact_prefill').update(rows=4),
        'wrong_fused_route':lambda d:event(d,'native_fused_mlp').update(blocks=0),
    }
    for name,mutate in mutations.items():
        bad=copy.deepcopy(data);mutate(bad);require(bad!=data,'ineffective mutation '+name)
        try:verify(bad,fixture,controls,phases)
        except (ValueError,KeyError,TypeError,IndexError,AssertionError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    total=sum(len(c['generated_ids']) for c in uncached.values())
    result=dict(passed=True,scope=__doc__,phases=reports,all_output_ids_and_text_exact_uncached=True,
        full_target_digest_pairs=9*2*65,hidden_values_compared=9*2*4096,output_tokens_per_arm=total,
        all_target_states_exact_uncached=True,negative_controls_rejected=rejected,files_sha256=files,
        frozen_native_control_prefixes_checked=True,three_survivors_ready_before_each_newcomer_native_plan=True,
        seed_client_arrival_order_not_assumed=True,initial_A_submission_does_not_prove_all_seeds_prefilled=True,
        prefix_cache_lineage_qualified=True,hybrid_recurrent_model_tested=False,coding_quality_qualified=False,
        survivor_witness_bound_to_actual_newcomer_tick=True,
        performance_measurement=False,serving_speed_claim=False,production_runtime_integrated=False,goal_complete=False)
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['files_sha256','phases']},indent=2))
if __name__=='__main__':main()
