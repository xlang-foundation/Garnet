"""Native B4 helper: serial-row control versus joint-row execution; correctness only."""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];mode=sys.argv[5];assert mode in ('serial','batch') and not os.path.exists(output)
assert inputs['batch']==4 and inputs['capacity']==512 and inputs['prefill_chunk']==16
with open(inputs['draft_model']+'/config.json') as f:config=json.load(f)
decoder=None;leases={};frames=[];calls=[];rejections=[];claims=[];cleanup=None;prepared=None;done=False;bank_created=False
def val(v,shape):return tensor.tensor(v,shape=shape,dtype=tensor.float32)
def state():
 out=[]
 for i in range(24):
  pair=decoder.session.snapshot(i)
  out.append([[decoder.io.verification_f32_sha256(decoder.io.rows(part,row,1)) for part in pair] for row in range(4)])
 hidden=None
 if bank_created:
  bank=decoder.session.diagnostic_final_hidden();hidden=[decoder.io.verification_f32_sha256(decoder.io.rows(bank,row,1)) for row in range(4)]
 return dict(kv=out,hidden=hidden)
def save():
 with open(output,'w') as f:json.dump(dict(completed=done,mode=mode,inputs=inputs,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,claims=claims,frames=frames,calls=calls,rejections=rejections,cleanup=cleanup,scope=__doc__),f,indent=2)
def claim(label):
 identity=len(claims)+1;lease=decoder.session.claim_request_row(identity);assert lease[0]>=0
 leases[label]=dict(row=lease[0],epoch=lease[1],request=identity);claims.append(dict(label=label,**leases[label]))
def epochs(labels):
 out=[0]*4
 for label in labels:out[leases[label]['row']]=leases[label]['epoch']
 return out
def prefill(queries,offset):
 global bank_created
 cursors=dict(offset);plans=[]
 while any(cursors[q['label']]<len(q['prompt_ids']) for q in queries):
  byrow={leases[q['label']]['row']:q for q in queries};ids=[];slots=[];flags=[];selected=[];lengths={}
  for row in range(4):
   q=byrow.get(row);length=min(16,len(q['prompt_ids'])-cursors[q['label']]) if q else 0
   if length:selected.append(q['label']);lengths[q['label']]=length
   for t in range(16):
    on=t<length;position=cursors[q['label']]+t if on else 0
    ids.append(q['prompt_ids'][position] if on else 0);slots.append(float(position));flags.append(float(on))
  ep=epochs(selected);hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*16+t] for row in range(4)]) for t in range(16)])
  decoder.session.run_tokens(graphs,hidden,val([s for s in slots for a in range(3)],[4,16,3]),val(slots,[4,16]),val(flags,[4,16]),-1,ep,ids)
  bank_created=True
  call=dict(stage='prefill',labels=selected,epochs=ep,tokens=ids,slots=slots,active=flags,offsets=dict(cursors),lengths=lengths);calls.append(call);plans.append(len(calls)-1)
  for label,length in lengths.items():cursors[label]+=length
 return plans
def propose(queries,offset):
 plans=prefill(queries,offset);labels=[q['label'] for q in queries]
 predicted=decoder.session.predict_rows(epochs(labels));tokens={q['label']:[] for q in queries};executed={q['label']:[] for q in queries};stop={}
 for step in range(8):
  advancing=[];ids=[0]*4;slots=[0.]*4;flags=[0.]*4
  for q in queries:
   label=q['label'];row=leases[label]['row']
   if label in stop:continue
   token=predicted[row]
   if token in inputs['draft_stops']:stop[label]='eos';continue
   tokens[label].append(token)
   if len(tokens[label])==q['draft_limit']:stop[label]='limit';continue
   ids[row]=token;slots[row]=float(len(q['prompt_ids'])+len(tokens[label])-1);flags[row]=1.;advancing.append(label);executed[label].append(token)
  if not advancing:break
  ep=epochs(advancing)
  predicted=decoder.session.run_tokens(base,decoder.io.embedding(ids),val([s for s in slots for a in range(3)],[4,3]),val(slots,[4]),val(flags,[4]),0,ep,ids)
  calls.append(dict(stage='decode',labels=advancing,epochs=ep,tokens=ids,slots=slots,active=flags))
 return dict(tokens=tokens,executed=executed,stop=stop,prefill_calls=plans)
try:
 decoder=DenseMetalDecoder(inputs['draft_model']+'/model.safetensors',config,4,512)
 decoder.enable_prefill(16);graphs=decoder._resident_prefill_graphs(16);base=[l['graph'] for l in decoder.layers];decoder.prefill_compile_ms=0
 decoder.session.configure_token_tracking([inputs['draft_revision'],decoder.io.checkpoint_format,inputs['draft_tokenizer_sha256'],'text_zero_based_3axis_v1','helper-B4-D512-T16-profile-v1','text'])
 tokenizer=garnet.tokenizer();tokenizer.open(inputs['draft_model'])
 for frame in inputs['frames']:
  for q in frame['queries']:assert tokenizer.encode_ex(q['prompt'],False)==q['prompt_ids']
 for label in inputs['initial']:claim(label)
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),prefill_graphs=len(decoder.prefill_graphs),checkpoint_format=decoder.io.checkpoint_format)
 assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-536870912
 print('helper4_ready',mode,prepared['memory']['metal_allocated_bytes'],flush=True)
 for index,frame in enumerate(inputs['frames']):
  before=state();released=None
  if frame.get('release'):
   label=frame['release'];released=leases.pop(label);assert decoder.session.release_request_row(released['row'],released['epoch']);claim(frame['claim']);assert leases[frame['claim']]['row']==released['row'] and leases[frame['claim']]['epoch']>released['epoch']
   snapshot=state();failed=False
   try:decoder.session.reconcile_dense_tokens(released['row'],released['epoch'],frame['queries'][0]['prompt_ids'])
   except Exception:failed=True
   assert failed and state()==snapshot;rejections.append(dict(kind='stale_replaced_epoch',frame=index,row=released['row']))
  controls=[];offset={}
  for q in frame['queries']:
   lease=leases[q['label']];retained=decoder.session.reconcile_dense_tokens(lease['row'],lease['epoch'],q['prompt_ids']);stats=json.loads(decoder.session.dense_reconcile_stats_json())
   assert stats['retained_tokens']==retained and stats['host_snapshot_bytes']==0
   offset[q['label']]=retained;controls.append(dict(label=q['label'],**lease,stats=stats))
  after_reconcile=state();results=[]
  for queries in ([frame['queries']] if mode=='batch' else [[q] for q in frame['queries']]):results.append(propose(queries,offset))
  after=state();frames.append(dict(index=index,leases=dict(leases),before=before,released=released,reconcile=controls,after_reconcile=after_reconcile,results=results,after=after))
  save();print('helper4_frame',mode,index,[len(q) for r in results for q in r['tokens'].values()],flush=True)
 assert decoder.prefill_compile_ms==0
 for label,lease in list(leases.items()):assert decoder.session.release_request_row(lease['row'],lease['epoch'])
 leases={};done=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 assert all(cleanup[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')) and cleanup['metal_allocated_bytes']<=4194304
 save()
