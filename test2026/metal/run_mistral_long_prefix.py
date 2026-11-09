"""Direct native long-prefix/cache-budget qualification. Hidden() is checked only after B4 decode; persistent survivor state uses complete KV/position/prediction during other frames. Serial fresh vs cached token-API arrivals; not HTTP TTFT or cold comparator admission."""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
lib,source,fixture_path,out,mode=sys.argv[1:];assert mode in ('control','candidate') and not os.path.exists(out);os.mkdir(out)
fixture=json.load(open(fixture_path));prompts=fixture['cases'][1]['prompts'];assert list(map(len,prompts))==[2048,2051,2048,2049]
config=json.load(open(fixture['model']+'/config.json'));decoder=None;complete=False;events=[];ready=None;cleanup=None;seed128=None;initial=None;initial_full=None
limit=(512 if mode=='control' else 3072)*1024*1024;epochs=[0]*4;pos=[0]*4;generated=[[] for _ in range(4)];next_id=100
identity=['mistral7b-a4b8f870-fp16g64','B4C4096T128-prefix-host-'+str(limit),'mistral-v03-native-tokenizer','text_zero_based_3axis_v1','long-prefix-budget-qualification','text']
def f32(a,s):return tensor.tensor(a,shape=s,dtype=tensor.float32)
def stats():return json.loads(decoder.session.token_prefix_stats_json())
def memory():return json.loads(garnet.memory_stats_json())
def save():
 with open(out+'/results.json','w') as f:json.dump(dict(completed=complete,scope=__doc__,mode=mode,library=garnet.native_library_path(),profile_host_budget=limit,ready=ready,seed128=seed128,initial=initial,initial_full=initial_full,events=events,cleanup=cleanup),f,indent=2)
def signatures(rows,hidden=False):
 result={str(row):dict(kv=[],hidden=None,prediction=None,position=pos[row]) for row in rows}
 for layer in range(32):
  parts=decoder.session.snapshot(layer)
  for row in rows:result[str(row)]['kv'].append([decoder.io.verification_f32_sha256(decoder.io.rows(p,row,1)) for p in parts])
 predicted=decoder.session.predict_rows([epochs[row] if row in rows else 0 for row in range(4)])
 for row in rows:result[str(row)]['prediction']=predicted[row]
 if hidden:
  h=decoder.session.hidden()
  for row in rows:result[str(row)]['hidden']=decoder.io.verification_f32_sha256(decoder.io.rows(h,row,1))
 return result
def prefill(rows,capture128=False):
 global seed128
 for offset in range(0,max(map(len,rows)),128):
  chunk=2
  while chunk<min(128,max(map(len,rows))-offset):chunk*=2
  counts=[min(chunk,max(0,len(p)-offset)) for p in rows];ids=[rows[row][offset+t] if t<counts[row] else 0 for row in range(4) for t in range(chunk)];slots=[pos[row]+t if t<counts[row] else 0 for row in range(4) for t in range(chunk)];active=[int(t<counts[row]) for row in range(4) for t in range(chunk)]
  h=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
  decoder.session.run_tokens(decoder.prefill_profile_graphs(chunk),h,f32([v for v in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]),-1,epochs,ids)
  assert decoder.session.affine_profile_misses==0
  pos[:]=[pos[row]+counts[row] for row in range(4)]
  if capture128 and offset==0:
   assert pos[0]==128;assert decoder.session.cache_token_prefix(0,epochs[0]);seed128=signatures([0])['0']
def decode7(first):
 generated[0]=[first]
 for tick in range(7):
  ids=[g[-1] for g in generated];predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),f32([v for p in pos for v in [p,p,p]],[4,3]),f32(pos,[4]),f32([1]*4,[4]),0,epochs,ids)
  for row in range(4):generated[row].append(predicted[row]);pos[row]+=1
  assert decoder.session.affine_profile_misses==0
 return [list(g) for g in generated]
def replace():
 global next_id
 before=signatures([1,2,3]);old=epochs[0];assert decoder.session.release_request_row(0,old);next_id+=1;lease=decoder.session.claim_request_row(next_id);assert lease[0]==0;epochs[0]=lease[1];pos[0]=0;generated[0]=[]
 assert signatures([1,2,3])==before
 return old,before
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_prefill(128)
 assert garnet.native_library_path()==lib+'/libgarnet.dylib';backend=garnet.backend_info();assert 'native_hybrid_token_prefix' in backend and 'mistral_grouped_value_context_two_heads' not in backend
 if mode=='candidate':assert 'native_prefix_host_limit_bytes=3221225472' in backend
 rejected=False
 try:decoder.session.configure_token_prefix(identity,16,limit+1)
 except:rejected=True
 assert rejected
 decoder.session.configure_token_prefix(identity,16,limit);ready=dict(backend=backend,memory=memory(),above_compiled_budget_rejected=True,identity=identity);decoder.reset()
 for row in range(4):next_id+=1;lease=decoder.session.claim_request_row(next_id);assert lease[0]==row;epochs[row]=lease[1]
 base=[l['graph'] for l in decoder.layers]
 prefill(prompts,True);predicted=list(decoder.session.predict_rows(epochs));initial_full=dict(kv=[[decoder.io.verification_f32_sha256(x) for x in decoder.session.snapshot(i)] for i in range(32)],hidden=decoder.io.verification_f32_sha256(decoder.session.hidden()))
 roots_prefill=signatures([0,1,2,3])
 before=signatures([0])['0'];before_stats=stats();start=time.monotonic();cached=decoder.session.cache_token_prefix(0,epochs[0]);capture_seconds=time.monotonic()-start;after=signatures([0])['0'];assert before==after and cached==(mode=='candidate')
 if mode=='control':assert stats()==before_stats
 all_roots=[cached]
 for row in [1,2,3]:
  kept=decoder.session.cache_token_prefix(row,epochs[row]);assert kept==(mode=='candidate');all_roots.append(kept)
 events.append(dict(kind='capture2048',stored=cached,all_four_stored=all_roots,roots_prefill=roots_prefill,seconds=capture_seconds,cache=stats(),state_unchanged=True,memory=memory()))
 for row in range(4):generated[row]=[predicted[row]]
 ids=decode7(predicted[0]);initial=dict(generated_ids=ids,signatures=signatures([0,1,2,3],True));save();print('long_prefix_initial',mode,cached,flush=True)
 for label,prompt in [('exact',prompts[0]),('appended',prompts[0]+initial['generated_ids'][0])]:
  old,before_survivors=replace();start=time.monotonic();retained=decoder.session.restore_token_prefix(0,epochs[0],prompt) if mode=='candidate' else 0
  assert retained==(2048 if mode=='candidate' else 0);pos[0]=retained
  if retained<len(prompt):prefill([prompt[retained:],[],[],[]])
  first=decoder.session.predict_rows([epochs[0],0,0,0])[0];seconds=time.monotonic()-start
  assert signatures([1,2,3])==before_survivors
  before=signatures([0])['0'];ids=decode7(first);after=signatures([0,1,2,3],True)
  if label=='exact':assert ids[0]==initial['generated_ids'][0] and after['0']==initial['signatures']['0']
  events.append(dict(kind='arrival',label=label,prompt_ids=prompt,lookup_free=mode=='control',retained=retained,native_first_prediction_seconds=seconds,before=before,generated_ids=ids,after=after,survivors_before=before_survivors,survivors_unchanged_before_decode=True,cache=stats(),memory=memory()));save();print('long_prefix_arrival',mode,label,retained,seconds,flush=True)
 old,before_survivors=replace();retained=decoder.session.restore_token_prefix(0,epochs[0],prompts[0][:1024]);assert retained==128;pos[0]=retained;short=signatures([0])['0'];assert short==seed128 and signatures([1,2,3])==before_survivors
 events.append(dict(kind='shorter',query_tokens=1024,retained=retained,signature=short,survivors_unchanged=True))
 old,before_survivors=replace();changed=list(prompts[0]);changed[0]=(changed[0]+17)%32768;retained=decoder.session.restore_token_prefix(0,epochs[0],changed);assert retained==0 and signatures([1,2,3])==before_survivors
 rejected=False
 try:decoder.session.restore_token_prefix(0,old,prompts[0])
 except:rejected=True
 assert rejected;events.append(dict(kind='changed-first-token',retained=0,stale_epoch_rejected=True,cache=stats(),memory=memory()))
 root_hits=[]
 if mode=='candidate':
  for row in [1,2,3]:
   others=[i for i in [1,2,3] if i!=row];survivor_before=signatures(others)
   old=epochs[row];assert decoder.session.release_request_row(row,old);next_id+=1;lease=decoder.session.claim_request_row(next_id);assert lease[0]==row;epochs[row]=lease[1];pos[row]=0
   kept=decoder.session.restore_token_prefix(row,epochs[row],prompts[row]);assert kept==len(prompts[row]);pos[row]=kept
   restored=signatures([row])[str(row)];assert restored==roots_prefill[str(row)] and signatures(others)==survivor_before
   root_hits.append(dict(row=row,retained=kept,restored=restored,survivors_unchanged=True))
 events.append(dict(kind='all-root-restores',checked=mode=='candidate',roots=root_hits,cache=stats(),memory=memory()))
 complete=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=memory() if decoder is not None else json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save()
