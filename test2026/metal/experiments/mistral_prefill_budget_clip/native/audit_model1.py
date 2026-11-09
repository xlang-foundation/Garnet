"""Audit full32-layer state and exact clipped native plans; no performance claim."""
from pathlib import Path
import collections,copy,hashlib,json,math,re,struct,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'model-audit1.json'
J=ROOT.parent/'mistral-prefill-completion-wide1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def events(p):
    result=[]
    for line in p.read_text().splitlines():
        try:e=json.loads(line)
        except ValueError:continue
        if isinstance(e,dict) and 'event' in e:result.append(e)
    return result
def select(es,kind):return [e for e in es if e['event']==kind]
def flatten(v):return [x for a in v for x in flatten(a)] if isinstance(v,list) else [v]
fixture=read(ROOT/'fixture-model1.json')
frozen=events(J/'model-control2.stdout');expected={e['id']:e for e in select(frozen,'completed')}
def verify(es,label):
    need(es[-1]['event']=='closed' and not select(es,'error'),'unclean model events')
    profile=read(ROOT/('profile-model-'+label+'1.json'));ready=select(es,'ready')
    need(len(ready)==1 and ready[0]['profile']==profile and 'half_attention_maximum_chunk=512' in ready[0]['backend'],'wrong model/backend profile')
    need(profile['prefill_completion_priority'] is (label=='clip') and profile['wide_owner_prefill'] is (label=='clip'),'wrong state comparison arm')
    need(select(es,'native_prefill_completion_policy')==[dict(event='native_prefill_completion_policy',enabled=(label=='clip'),owner_limit=1 if label=='clip' else 4,chunk=128)],'wrong configured native policy')
    done=select(es,'completed');need(len(done)==4 and {e['id'] for e in done}==set(expected),'wrong completed fixtures')
    outputs={e['id']:e for e in done};native={e['native_request']:e for e in done}
    for req in fixture['requests']:
        key='trained-'+str(req['row']);actual=outputs[key];saved=expected[key]
        for field in ['prompt_ids','cached_tokens','finish_reason']:need(actual[field]==saved[field],'frozen output differs '+key+' '+field)
        need(actual['prompt_ids']==req['token_ids'] and len(actual['generated_ids'])==32 and actual['generated_ids'][:8]==saved['generated_ids'] and actual['cached_tokens']==0,'wrong prompt/output/cache bounds')
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
        if label=='clip' and prefill:need(len(b['requests'])==1 and chunk in [2,4,8,16,32,64,128,256,512],'wrong prepared owner profile')
    witnesses=select(es,'diagnostic_request_state');need(len(witnesses)==8,'missing per-request states')
    keys={(e['id'],e['phase']) for e in witnesses};need(keys=={(key,phase) for key in outputs for phase in ['prefill','final']},'duplicate/missing state phases')
    for w in witnesses:
        out=outputs[w['id']];need((w['native_request'],w['row'],w['epoch'])==(out['native_request'],out['native_row'],out['native_epoch']),'state belongs to another lease')
        need(w['consumed_tokens']==len(out['prompt_ids'])+(31 if w['phase']=='final' else 0),'wrong witnessed prefix extent')
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
    return dict(requests=4,tokens=128,per_request_states=8,per_request_digest_count=520,hidden_values=32768,fused_blocks=sum(e['blocks'] for e in fusion),half_dispatches=sum(e['dispatches'] for e in half),chunks=dict(collections.Counter(e['prefill_chunk'] for e in batches if e['stage']=='prefill')),postclose_metal_bytes=close['metal_allocated_bytes'])

def compare(control,clip):
    result={label:verify(es,label) for label,es in [('control',control),('clip',clip)]}
    need(select(clip,'native_prefill_budget_clip_policy')==[dict(event='native_prefill_budget_clip_policy',enabled=True,maximum_extent=512,actual_global_budget=512)],'missing native clip configuration')
    need('native_prefill_budget_clip_optional_v1' in select(clip,'ready')[0]['backend'],'missing native backend option')
    by=lambda es:{(e['id'],e['phase']):e for e in select(es,'diagnostic_request_state')}
    before,after=by(control),by(clip)
    for key in before:
        for field in ['consumed_tokens','kv','hidden','hidden_values']:
            need(before[key][field]==after[key][field],'schedule changed complete request state '+str(key)+' '+field)
    # The extended control must still reproduce frozen eight-token prefixes
    # and complete per-row state immediately after each original prompt.
    old=by(frozen)
    for key in old:
        if key[1]=='prefill':
            for field in ['consumed_tokens','kv','hidden','hidden_values']:
                need(before[key][field]==old[key][field],'new control changed frozen prompt state '+str(key))
    outputs=lambda es:{e['id']:e for e in select(es,'completed')}
    c,w=outputs(control),outputs(clip)
    for key in c:
        for field in ['prompt_ids','generated_ids','text','cached_tokens','finish_reason']:
            need(c[key][field]==w[key][field],'new complete output differs '+key+' '+field)
    ticks=select(clip,'native_profile_tick');need(ticks,'missing native profile ticks')
    flattened=[];partial=collections.Counter();maximum_actual=0
    for tick in ticks:
        batches=tick['batches'];decode=[b for b in batches if b['stage']=='decode'];prefill=[b for b in batches if b['stage']=='prefill']
        need(len(decode)<=1 and len(prefill)<=1 and len(batches)==len(decode)+len(prefill),'unexpected tick stages')
        if decode:need(batches[0]['stage']=='decode','prefill ran ahead of ready decode')
        used=sum(len(b['requests']) for b in decode);budget=512-used
        for batch in prefill:
            need(len(batch['requests'])==1,'clipped prefill changed owner cardinality')
            q=batch['requests'][0];remaining=q['prompt_tokens']-q['prefill_offset']
            bound=min(remaining,budget,512);shape=1<<max(1,(bound-1).bit_length())
            count=min(remaining,shape,budget)
            need(bound>0 and batch['prefill_chunk']==shape and q['prefill_tokens']==count,'actual quantum or up-to prepared extent differs from residual budget')
            used+=count
            if count in [509,510,511] and shape==512:partial[count]+=1
        need(used==tick['token_count'] and used<=512,'actual tick budget disagrees with native packed plan')
        maximum_actual=max(maximum_actual,used)
        flattened.extend(batches)
    batches=select(clip,'native_batch');need(len(flattened)==len(batches),'profile tick/body execution count mismatch')
    for plan,body in zip(flattened,batches):
        need(plan['stage']==body['stage'] and plan['requests']==body['requests'],'body differs from native scheduled plan')
    need(set(partial)=={509,510,511},'all three live-decoder residual budgets were not executed')
    result['clipped_actual_count_histogram']=dict(partial);result['maximum_actual_tick_tokens']=maximum_actual
    return result

def main():
    need(not DEST.exists(),'never overwrite audit')
    pre=read(ROOT/'model1-preflight.json');post=read(ROOT/'model1-postflight.json')
    need(pre['passed'] and pre['performance_measurement'] is False and post['passed'] and post['all_owned_processes_terminal'],'unfinished state collection')
    for name,h in pre['files_sha256'].items():need(sha(Path(name))==h,'changed state input '+name)
    status=[read(ROOT/('model-'+label+'1-status.json')) for label in ['control','clip']]
    need(all(s['status']=='terminal' and s['returncode']==0 for s in status) and status[1]['started_ns']>=status[0]['finished_ns'] and status[0]['pid']!=status[1]['pid'],'engines were not serial/fresh/completed')
    control=events(ROOT/'model-control1.stdout');clip=events(ROOT/'model-clip1.stdout');report=compare(control,clip)
    def event(es,kind):return select(es,kind)[0]
    mutations={
        'wrong_kv':lambda c,w:event(w,'diagnostic_request_state')['kv'][0].__setitem__(0,'0'*64),
        'wrong_hidden':lambda c,w:event(w,'diagnostic_request_state').update(hidden='0'*64),
        'wrong_hidden_values':lambda c,w:event(w,'diagnostic_request_state').update(hidden_values=[[0.]*4096]),
        'stale_state_epoch':lambda c,w:event(w,'diagnostic_request_state').update(epoch=0),
        'wrong_consumed_extent':lambda c,w:event(w,'diagnostic_request_state').update(consumed_tokens=1),
        'wrong_output':lambda c,w:event(w,'completed')['generated_ids'].__setitem__(10,-1),
        'missing_hoist':lambda c,w:event(w,'diagnostic_native_kv_hoist').update(dispatches=0),
        'missing_state':lambda c,w:w.remove(event(w,'diagnostic_request_state')),
        'changed_control_state':lambda c,w:event(c,'diagnostic_request_state')['kv'][0].__setitem__(0,'0'*64),
        'owner_leak':lambda c,w:event(w,'closed')['memory'].update(garnet_live_matrix_owners=1),
        'actual_tick_overflow':lambda c,w:event(w,'native_profile_tick').update(token_count=513),
        'unprepared_extent':lambda c,w:next(b for t in select(w,'native_profile_tick') for b in t['batches'] if b['stage']=='prefill').update(prefill_chunk=1024),
        'wrong_clipped_count':lambda c,w:next(b for t in select(w,'native_profile_tick') for b in t['batches'] if b['stage']=='prefill')['requests'][0].update(prefill_tokens=513),
        'missing_clip_policy':lambda c,w:w.remove(event(w,'native_prefill_budget_clip_policy')),
    }
    rejected=[]
    for name,mutate in mutations.items():
        c,w=copy.deepcopy(control),copy.deepcopy(clip);mutate(c,w)
        try:compare(c,w)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    result=dict(passed=True,scope=__doc__,qualification=report,all128_output_ids_and_text_exact_control=True,all_complete_request_kv_hidden_states_exact=True,per_request_state_digests_compared=520,per_request_hidden_values_compared=32768,frozen_prefill_state_digests_compared=260,frozen_first8_ids_per_request_exact=True,negative_controls_rejected=rejected,preflight_sha256=sha(ROOT/'model1-preflight.json'),stdout_sha256={label:sha(ROOT/('model-'+label+'1.stdout')) for label in ['control','clip']},performance_measurement=False,prefix_cache_reuse_qualified=False,coding_quality_qualified=False,production_runtime_integrated=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
