# SPDX-License-Identifier: Apache-2.0
"""B4 streaming, native token-prefix snapshots and continuous A-to-E row reuse.

Args LIB SOURCE FIXTURE NEW_OUTPUT control|candidate. Direct token API episodes
use fixed8-output budgets (including EOS if returned); no chat/coding admission.
Fresh row execution checks complete KV and current decode hidden for matching
precision histories. Cold FP16 equivalence of decoded snapshots is recorded
separately and is NOT admitted. Prefix
hidden restoration checked by native next-token prediction. Scheduler ticks
execute separate prefill/decode commands, not a fused mixed forward. No timing
claim is made for this correctness campaign. All inference is serial.
"""
import sys,os,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
out=sys.argv[4];mode=sys.argv[5];assert mode in ('control','candidate') and not os.path.exists(out);os.mkdir(out)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib' and 'mistral_packed_prefill_optional' in garnet.backend_info()
with open(fixture['model']+'/config.json') as f:config=json.load(f)
long=fixture['cases'][1]['prompts'];prompts=[list(p[:512]) for p in long];assert len(set(tuple(p[:128]) for p in prompts))==4
a=prompts[0];e=list(a);e[0]=17;changed=list(a);changed[300]=(changed[300]+17)%32768
identity=['mistral7b-a4b8f870-fp16g64','B4C4096T128-packed512-optional','mistral-v03-native-tokenizer','text_zero_based_3axis_v1','packed-prefix-qualification','text']
decoder=None;policy=None;ready=None;cleanup=None;finished=False;frames=[];leases=[];captures=[];restores=[];rejected=[];episodes=[];fresh=[];survivors=[];stability=[];policy_fresh=[];scheduler=None
pos=[0]*4;epochs=[0]*4;g=[[] for _ in range(4)];next_request=100

def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def save():
 with open(out+'/results.json','w') as f:json.dump(dict(completed=finished,scope=__doc__,mode=mode,library=garnet.native_library_path(),backend=garnet.backend_info(),fixture=sys.argv[3],ready=ready,identity=identity,host_cache_bytes=384*1024*1024,cache_entries=16,frames=frames,leases=leases,captures=captures,restores=restores,rejected=rejected,episodes=episodes,fresh=fresh,survivors=survivors,stability=stability,policy_fresh=policy_fresh,scheduler=scheduler,cleanup=cleanup),f,indent=2)
def kv(rows):
 result=[[] if row in rows else None for row in range(4)]
 for layer in range(32):
  parts=decoder.session.snapshot(layer)
  for row in rows:result[row].append([decoder.io.verification_f32_sha256(decoder.io.rows(p,row,1)) for p in parts])
 return result

def signature(row,decode=False):
 mask=[epochs[i] if i==row else 0 for i in range(4)]
 return dict(kv=kv([row])[row],prediction=decoder.session.predict_rows(mask)[row],hidden_sha256=decoder.io.verification_f32_sha256(decoder.io.rows(decoder.session.hidden(),row,1)) if decode else None,position=pos[row])
def reset():
 decoder.reset();pos[:]=[0]*4;epochs[:]=[0]*4;g[:]=[[] for _ in range(4)]
def claim(row):
 global next_request
 next_request+=1;lease=decoder.session.claim_request_row(next_request);assert lease[0]==row;epochs[row]=lease[1];pos[row]=0;g[row]=[]
 leases.append(dict(request=next_request,row=row,epoch=lease[1]));return lease[1]
def stable(label,call):
 before=kv([1,2,3]);call();after=kv([1,2,3]);assert before==after
 record=dict(label=label,before=before[1:],after=after[1:]);stability.append(record);return record
def replace():
 old=epochs[0]
 def change():
  assert decoder.session.release_request_row(0,old);epochs[0]=0;claim(0)
 check=stable('release-and-reclaim',change);assert all(h==zero_hash for pair in kv([0])[0] for h in pair)
 return old,check

def prefill(rows,tag,force_chunk=None):
 maxlen=max([len(p) for p in rows]+[0])
 for offset in range(0,maxlen,128):
  remaining=min(128,maxlen-offset);chunk=2
  while chunk<remaining:chunk*=2
  if force_chunk is not None:chunk=force_chunk
  counts=[min(chunk,max(0,len(p)-offset)) for p in rows]
  ids=[rows[row][offset+t] if t<counts[row] else 0 for row in range(4) for t in range(chunk)]
  slots=[pos[row]+t if t<counts[row] else 0 for row in range(4) for t in range(chunk)];active=[int(t<counts[row]) for row in range(4) for t in range(chunk)]
  hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
  before=list(pos);decoder.session.run_tokens(decoder.prefill_profile_graphs(chunk),hidden,f32([v for v in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]),-1,epochs,ids)
  expected=160 if mode=='candidate' and sum(counts)>=128 else 0
  assert decoder.session.packed_prefill_dispatches==expected and decoder.session.affine_profile_misses==0
  pos[:]=[pos[row]+counts[row] for row in range(4)]
  frames.append(dict(tag=tag,stage='prefill',chunk=chunk,counts=counts,before=before,after=list(pos),epochs=list(epochs),packed_dispatches=decoder.session.packed_prefill_dispatches))
  assert len(decoder.prefill_graphs)==ready['graphs'] and decoder.prefill_compile_ms==0 and decoder.session.packed_prefill_plans==3

def decode(ids,active,tag):
 slots=[pos[row] if active[row] else 0 for row in range(4)];before=list(pos)
 predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),f32([v for v in slots for _ in range(3)],[4,3]),f32(slots,[4]),f32(active,[4]),0,[epochs[row] if active[row] else 0 for row in range(4)],ids)
 assert decoder.session.packed_prefill_dispatches==0 and decoder.session.affine_profile_misses==0
 pos[:]=[pos[row]+active[row] for row in range(4)]
 frames.append(dict(tag=tag,stage='decode',active=list(active),before=before,after=list(pos),epochs=list(epochs),packed_dispatches=0))
 return predicted

def generate(label,prompt,first,budget=8,with_survivors=True):
 pre=signature(0);assert pre['position']==len(prompt) and pre['prediction']==first
 g[0]=[];start=[len(p) for p in g]
 for step in range(budget):
  active=[int(step>0)]+[int(with_survivors and bool(g[row])) for row in range(1,4)]
  predicted=decode([g[row][-1] if active[row] else 0 for row in range(4)],active,label) if any(active) else [-1]*4
  for row in range(4):
   if active[row] or row==0:g[row].append(predicted[row] if active[row] else first)
 end=signature(0,True)
 record=dict(label=label,prompt_ids=list(prompt),generated_ids=list(g[0]),epoch=epochs[0],prefill=pre,decode=end,survivor_before=start[1:],survivor_after=[len(p) for p in g[1:]],cache=json.loads(decoder.session.token_prefix_stats_json()))
 episodes.append(record);save();print('packed_prefix_episode',mode,label,pos[0],flush=True);return record

def capture(boundary):
 assert pos[0]==boundary and decoder.session.cache_token_prefix(0,epochs[0]);captures.append(dict(tokens=boundary,epoch=epochs[0],cache=json.loads(decoder.session.token_prefix_stats_json())))
def restore(prompt,wanted,label):
 def apply():
  kept=decoder.session.restore_token_prefix(0,epochs[0],prompt);assert kept==wanted,(label,kept,wanted);pos[0]=kept
 check=stable('restore-'+label,apply);restores.append(dict(label=label,epoch=epochs[0],prompt_ids=list(prompt),retained=wanted,survivor_check=check,cache=json.loads(decoder.session.token_prefix_stats_json())))
def reject(label,call):
 before=kv([0,1,2,3]);stats=decoder.session.token_prefix_stats_json();bad=False
 try:call()
 except Exception:bad=True
 assert bad and kv([0,1,2,3])==before and decoder.session.token_prefix_stats_json()==stats
 rejected.append(label)

try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.session.prefill_set_packed_affine(mode=='candidate');decoder.prepare_prefill(128);decoder.prefill_compile_ms=0;base=[x['graph'] for x in decoder.layers]
 assert decoder.session.configure_token_prefix(identity,16,384*1024*1024)
 ready=dict(graphs=len(decoder.prefill_graphs),plans=decoder.session.packed_prefill_plans,memory=json.loads(garnet.memory_stats_json()));assert ready['graphs']==224 and ready['plans']==3
 zero_hash=decoder.io.verification_f32_sha256(decoder.io.zeros([1,8,4096,128]))
 print('packed_prefix_ready',mode,ready['memory']['metal_allocated_bytes'],flush=True)
 claim(0);prefill([a[:64],[],[],[]],'seed64');capture(64);reset()
 for row in range(4):claim(row)
 prefill([p[:128] for p in prompts],'initial128');capture(128)
 prefill([p[128:] for p in prompts],'initial512');capture(512)
 first=decoder.session.predict_rows(epochs)
 for row in range(1,4):g[row]=[first[row]]
 initial=generate('A-initial',a,first[0]);old,check=replace()
 reject('stale-restore',lambda:decoder.session.restore_token_prefix(0,old,a));reject('stale-capture',lambda:decoder.session.cache_token_prefix(0,old))
 reject('uninitialized-prediction',lambda:decoder.session.predict_rows([epochs[0],0,0,0]))
 reject('late-profile-preparation',lambda:decoder.session.prepare_prefill(decoder.prefill_profile_graphs(128)[0],0))
 restore(e,0,'E-miss');check=stable('E-prefill',lambda:prefill([e,[],[],[]],'E'))
 generate('E-replacement',e,decoder.session.predict_rows([epochs[0],0,0,0])[0])
 replace();restore(a,512,'A-exact');exact=generate('A-exact',a,decoder.session.predict_rows([epochs[0],0,0,0])[0]);assert exact['generated_ids']==initial['generated_ids'] and exact['prefill']==initial['prefill'] and exact['decode']==initial['decode']
 appended=a+initial['generated_ids'];replace();restore(appended,512,'A-appended');stable('append-prefill',lambda:prefill([appended[512:],[],[],[]],'append'))
 generate('A-appended',appended,decoder.session.predict_rows([epochs[0],0,0,0])[0])
 replace();restore(changed,128,'A-changed');stable('changed-prefill',lambda:prefill([changed[128:],[],[],[]],'changed'))
 generate('A-changed',changed,decoder.session.predict_rows([epochs[0],0,0,0])[0])
 shorter=a[:256];replace();restore(shorter,128,'A-shorter');stable('shorter-prefill',lambda:prefill([shorter[128:],[],[],[]],'shorter'))
 generate('A-shorter',shorter,decoder.session.predict_rows([epochs[0],0,0,0])[0])
 for row in range(1,4):survivors.append(dict(row=row,prompt_ids=prompts[row],generated_ids=list(g[row]),signature=signature(row,True)))
 assert all(len(x['generated_ids'])==49 for x in survivors)
 # Fresh, lookup-free execution replays each request in a zeroed physical row.
 targets=[episodes[0]]+episodes[1:2]+episodes[3:]+survivors
 for index,target in enumerate(targets):
  reset();claim(0);prompt=target['prompt_ids'];prefill([prompt,[],[],[]],'fresh-'+str(index));prediction=decoder.session.predict_rows([epochs[0],0,0,0])[0]
  pre=signature(0);generated=[prediction]
  while len(generated)<len(target['generated_ids']):generated.append(decode([generated[-1],0,0,0],[1,0,0,0],'fresh-'+str(index))[0])
  end=signature(0,True);want=target.get('decode',target.get('signature'))
  assert generated==target['generated_ids'] and end==want,(index,'fresh row divergence')
  if 'prefill' in target:assert pre==target['prefill']
  fresh.append(dict(target=target.get('label','survivor-'+str(target.get('row'))),generated_ids=generated,prefill=pre,decode=end));save();print('packed_prefix_fresh',mode,index,'PASS',flush=True)
 # Separate native scheduler episode: queue E, recycle A, then match exact A
 # under a new scheduler request ID and append its generated history.
 reset();decoder.session.clear_token_prefixes();policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16) and policy.set_prefill_chunk(128)
 jobs={};done={};ticks=[];plans=[];mixed=0;head_only=0;stale_checked=False
 small=[p[:128] for p in prompts];replacement=list(small[0]);replacement[0]=17
 def submit(label,prompt,budget):
  identity=policy.submit_tokens(prompt,budget,0);assert identity>0;jobs[identity]=dict(label=label,prompt=list(prompt),budget=budget,generated=[]);return identity
 for row in range(4):submit('A' if row==0 else 'BCD'[row-1],small[row],2 if row==0 else 32)
 submit('E',replacement,2);pending_old=None
 while jobs:
  tick=json.loads(policy.next_json());assert tick['batches'];ticks.append(tick)
  if [p['stage'] for p in tick['batches']]==['decode','prefill']:mixed+=1
  for plan in tick['batches']:
   stage=plan['stage'];chunk=128 if stage=='prefill' else 1;requests={r['row']:r for r in plan['requests']};ids=[];slots=[];active=[];final=[0]*4
   for row in range(4):
    req=requests.get(row);count=(req['prefill_tokens'] if stage=='prefill' else 1) if req else 0
    for t in range(chunk):
     valid=t<count;slot=req['slot']+t if valid else 0;ids.append((jobs[req['id']]['prompt'][slot] if stage=='prefill' else req['token_id']) if valid else 0);slots.append(slot);active.append(int(valid))
    if req and (stage=='decode' or req['prefill_offset']+req['prefill_tokens']==req['prompt_tokens']):final[row]=req['epoch']
   if stage=='prefill':
    if any(active):
     hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
     decoder.session.run_tokens(decoder.prefill_profile_graphs(chunk),hidden,f32([v for v in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]),-1,plan['epochs'],ids)
     packed=decoder.session.packed_prefill_dispatches;assert packed==(160 if mode=='candidate' and sum(active)>=128 else 0)
    else:packed=0;head_only+=1
    predicted=decoder.session.predict_rows(final) if any(final) else None
   else:
    predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),f32([v for v in slots for _ in range(3)],[4,3]),f32(slots,[4]),f32(active,[4]),0,plan['epochs'],ids);packed=decoder.session.packed_prefill_dispatches;assert packed==0
   plans.append(dict(stage=stage,requests=plan['requests'],epochs=plan['epochs'],active_tokens=sum(active),packed_dispatches=packed,head_only=not any(active)))
   for row,req in requests.items():
    job=jobs[req['id']]
    if not final[row]:assert policy.complete_prefill_chunk(req['id'],req['prefill_tokens'])
    else:
     token=predicted[row];job['generated'].append(token);stop=len(job['generated'])==job['budget']
     state=kv([row])[row] if stop else None
     hidden=decoder.io.verification_f32_sha256(decoder.io.rows(decoder.session.hidden(),row,1)) if stop else None
     fn=policy.complete_prefill if stage=='prefill' else policy.complete_decode;assert fn(req['id'],token,stop)==(3 if stop else 2)
     if stop:
      done[job['label']]=dict(label=job['label'],request_id=req['id'],row=row,epoch=req['epoch'],prompt_ids=job['prompt'],generated_ids=list(job['generated']),state_sha256=state,hidden_sha256=hidden);del jobs[req['id']]
      if job['label']=='E':submit('A-return',small[0],2)
      if job['label']=='A-return':submit('A-append',small[0]+done['A']['generated_ids'],2)
   if stage=='decode' and pending_old is None:pending_old=(list(plan['epochs']),list(ids),list(slots),list(active))
  if not stale_checked and 'E' in done:
   oldep,oldids,oldslots,oldactive=pending_old
   reject('scheduler-stale-dispatch',lambda:decoder.session.run_leased(base,decoder.io.embedding(oldids),f32([v for v in oldslots for _ in range(3)],[4,3]),f32(oldslots,[4]),f32(oldactive,[4]),0,oldep));stale_checked=True
 assert mixed>=2 and head_only>=1 and stale_checked and set(done)=={'A','B','C','D','E','A-return','A-append'}
 assert done['A']['generated_ids']==done['A-return']['generated_ids']
 assert len({done[k]['epoch'] for k in ['A','E','A-return','A-append']})==4 and all(done[k]['row']==0 for k in ['A','E','A-return','A-append'])
 assert not json.loads(policy.next_json())['batches'];scheduler=dict(records=done,ticks=ticks,plans=plans,mixed_ticks=mixed,head_only=head_only,cache=json.loads(decoder.session.token_prefix_stats_json()),status=json.loads(policy.status_json()));policy.close();policy=None
 for label in ['A','B','C','D','E','A-append']:
  target=done[label];reset();claim(0);prefill([target['prompt_ids'],[],[],[]],'scheduler-fresh-'+label)
  generated=[decoder.session.predict_rows([epochs[0],0,0,0])[0]]
  while len(generated)<len(target['generated_ids']):generated.append(decode([generated[-1],0,0,0],[1,0,0,0],'scheduler-fresh-'+label)[0])
  state=signature(0,True);cold=None;recipe=dict(prefill_tokens=len(target['prompt_ids']),fp32_decode_tokens=0,remaining_prefill_tokens=0)
  if label=='A-append':
   # Preserve the failed cold full-state comparison, then independently
   # replay the provenance that actually produced the stored129 boundary.
   cold=dict(generated_ids=list(generated),signature=state,state_equal=state['kv']==target['state_sha256'] and state['hidden_sha256']==target['hidden_sha256'])
   assert generated==target['generated_ids']
   reset();claim(0);prefill([target['prompt_ids'][:128],[],[],[]],'scheduler-lineage-A-append')
   assert decoder.session.predict_rows([epochs[0],0,0,0])[0]==target['prompt_ids'][128]
   assert decode([target['prompt_ids'][128],0,0,0],[1,0,0,0],'scheduler-lineage-A-append')[0]==target['prompt_ids'][129]
   prefill([target['prompt_ids'][129:],[],[],[]],'scheduler-lineage-A-append',128)
   generated=[decoder.session.predict_rows([epochs[0],0,0,0])[0]]
   while len(generated)<len(target['generated_ids']):generated.append(decode([generated[-1],0,0,0],[1,0,0,0],'scheduler-lineage-A-append')[0])
   state=signature(0,True);recipe=dict(prefill_tokens=128,fp32_decode_tokens=1,remaining_prefill_tokens=1,remaining_prefill_chunk=128)
  assert generated==target['generated_ids'] and state['kv']==target['state_sha256'] and state['hidden_sha256']==target['hidden_sha256'],('scheduler matching-lineage divergence',label)
  if label=='A':assert target['generated_ids']==done['A-return']['generated_ids'] and target['state_sha256']==done['A-return']['state_sha256'] and target['hidden_sha256']==done['A-return']['hidden_sha256']
  policy_fresh.append(dict(label=label,generated_ids=generated,signature=state,precision_recipe=recipe,cold_fp16=cold));save();print('packed_policy_fresh',mode,label,'PASS',flush=True)
 finished=True;print('packed_prefix_complete',mode,len(fresh),'fresh controls',mixed,'mixed ticks',flush=True)
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
