# SPDX-License-Identifier: Apache-2.0
"""Actual-weight dense-profile checks; deliberately not a serving speed test.

XLang arguments: LIB SOURCE FIXTURE CODING_CONTROL CONTEXT_CONTROL OUTPUT
BATCH CHUNK KIND POLICY. KIND is coding/context/all; POLICY is full/staggered.
Controls are the frozen batch-one native chunk16 collections, including their
known differences from MLX. Host final-hidden gathering is diagnostic transport.
Run only after the sequential comparator campaign and its engine groups exit.
"""
import sys, json, os, time
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])
import garnet, tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder

fixture_path=sys.argv[3];control_roots=dict(coding=sys.argv[4],context=sys.argv[5])
output=sys.argv[6];batch=int(sys.argv[7]);maximum=int(sys.argv[8])
kind=sys.argv[9];policy=sys.argv[10];capacity=4096
assert batch in (1,2,4) and maximum in (2,4,8,16,32,64,128,256,512)
assert batch*maximum<=512 and kind in ('coding','context','all') and policy in ('full','staggered')
assert policy=='full' or batch>1
assert not os.path.exists(output),'new evidence directory required'
with open(fixture_path) as stream:fixture=json.load(stream)
cases=[case for case in fixture['cases'] if kind=='all' or case['kind']==kind]
assert cases and all(0<len(case['prompt_ids']) and len(case['prompt_ids'])+case['max_tokens']<=capacity for case in cases)
controls={}
for name in ('coding','context'):
 with open(control_roots[name]+'/status.json') as stream:status=json.load(stream)
 assert status['completed'] and status['batch']==1 and status['capacity']==capacity and status['chunk']==16
for case in cases:
 with open(control_roots[case['kind']]+'/'+case['id']+'/request-0.result.json') as stream:controls[case['id']]=json.load(stream)
 assert controls[case['id']]['status']=='completed' and controls[case['id']]['prompt_ids']==case['prompt_ids']
os.mkdir(output)
with open(fixture['model']+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(fixture['model'])
for case in cases:assert tokenizer.encode(case['rendered'])==case['prompt_ids'],'native prompt IDs differ'
decoder=None;closed=None;records=[];prepared=None

def save(completed):
 with open(output+'/status.json','w') as stream:
  json.dump(dict(completed=completed,backend=garnet.backend_info(),fixture=fixture_path,
    native_library_root=sys.argv[1],source_root=sys.argv[2],
    loaded_native_library=garnet.native_library_path(),
    control_roots=control_roots,capacity=capacity,batch=batch,chunk=maximum,kind=kind,
    policy=policy,records=records,prepared=prepared,cleanup=closed,
    scope='Actual-weight profile/control parity and ragged row positions; host hidden gathering; no serving, speed, continuous-admission or quality claim'),stream,indent=2)

try:
 started=time.monotonic()
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,batch,capacity,resident=True)
 assert decoder.dense_prefill_limit==512 and 'prepared_dense_prefill' in garnet.backend_info()
 decoder.prepare_prefill(maximum)
 prepared_count=len(decoder.prefill_graphs)
 prepared=dict(initialization_seconds=time.monotonic()-started,graph_count=prepared_count,
    memory=json.loads(garnet.memory_stats_json()))
 print('mistral_profile_ready',json.dumps(prepared),flush=True)
 for start in range(0,len(cases),batch):
  # Repeat the selected tail group to exercise every physical row; identity and
  # row number are recorded independently, so repeats are never unique cases.
  group=cases[start:start+batch]
  while len(group)<batch:group.append(group[len(group)-1])
  decoder.reset();lengths=[len(case['prompt_ids']) for case in group]
  budgets=[min(8,case['max_tokens']) if policy=='staggered' and row==0 else case['max_tokens'] for row,case in enumerate(group)]
  final_hidden=[None for row in range(batch)];offset=0;frames=[];began=time.monotonic()
  while offset<max(lengths):
   chunk=decoder._prefill_chunk_size(max(lengths)-offset,maximum)
   graphs=decoder._resident_prefill_graphs(chunk)
   valid=[min(chunk,max(0,length-offset)) for length in lengths]
   before=list(decoder.slots)
   values=[decoder.io.embedding([group[row]['prompt_ids'][offset+i] if i<valid[row] else 0 for row in range(batch)]) for i in range(chunk)]
   hidden=decoder.io.stack_tokens(values)
   active=tensor.tensor([1 if i<valid[row] else 0 for row in range(batch) for i in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
   positions=tensor.tensor([offset+i if i<valid[row] else 0 for row in range(batch) for i in range(chunk) for axis in range(3)],shape=[batch,chunk,3],dtype=tensor.float32)
   slots=tensor.tensor([offset+i if i<valid[row] else 0 for row in range(batch) for i in range(chunk)],shape=[batch,chunk],dtype=tensor.float32)
   decoder.session.run(graphs,hidden,positions,slots,active,-1)
   assert decoder.session.kv_inplace_updates==64,'prefill did not update both private caches in all32 layers'
   completed=[row for row in range(batch) if valid[row] and offset+valid[row]==lengths[row]]
   if completed:
    flat=decoder.session.hidden().tolist()
    assert len(flat)==batch*chunk*4096,'unexpected final hidden storage'
    for row in completed:
     begin=(row*chunk+valid[row]-1)*4096
     final_hidden[row]=flat[begin:begin+4096]
   decoder.slots=[before[row]+valid[row] for row in range(batch)]
   assert len(decoder.prefill_graphs)==prepared_count,'request created a graph family'
   frames.append(dict(chunk=chunk,offset=offset,valid=valid,slots_before=before,slots_after=list(decoder.slots),
      kv_inplace_updates=decoder.session.kv_inplace_updates))
   offset+=chunk
   print('mistral_profile_prefill',start,offset,valid,flush=True)
  assert all(value is not None for value in final_hidden) and decoder.slots==lengths
  last=tensor.tensor([value for row in final_hidden for value in row],shape=[batch,4096],dtype=tensor.float32)
  logits=decoder.head.run([last]);first_logits=logits.tolist()
  prediction=decoder.io.argmax(decoder.io.logits(logits))
  generated=[[] for row in range(batch)];active=[1 for row in range(batch)];decode_frames=[]
  for step in range(max(budgets)):
   for row in range(batch):
    if active[row]:
     generated[row].append(prediction[row])
     if prediction[row] in fixture['stops'] or len(generated[row])==budgets[row]:active[row]=0
   if not any(active):break
   before=list(decoder.slots)
   prediction=decoder.step([prediction[row] if active[row] else 0 for row in range(batch)],active=active)
   assert decoder.slots==[before[row]+active[row] for row in range(batch)],'inactive row advanced'
   assert decoder.session.kv_inplace_updates==64,'decode did not update both private caches in all32 layers'
   decode_frames.append(dict(active=list(active),slots_before=before,slots_after=list(decoder.slots),
      kv_inplace_updates=decoder.session.kv_inplace_updates))
   if (step+1)%128==0:print('mistral_profile_decode',start,step+1,flush=True)
  rows=[]
  for row,case in enumerate(group):
   expected=controls[case['id']]['generated_ids'][:budgets[row]]
   ids=generated[row];common=0
   for a,b in zip(ids,expected):
    if a!=b:break
    common+=1
   assert decoder.slots[row]==lengths[row]+len(ids)-1
   rows.append(dict(id=case['id'],row=row,prompt_ids=case['prompt_ids'],budget=budgets[row],
     generated_ids=ids,text=tokenizer.decode_ex(ids,True),slots=decoder.slots[row],
     finish_reason='stop' if ids[-1] in fixture['stops'] else 'length',
     exact_native_control=ids==expected,matching_prefix_tokens=common))
  record=dict(group=start//batch,rows=rows,prefill_frames=frames,decode_frames=decode_frames,
     first_logits=first_logits,elapsed_seconds=time.monotonic()-began,
     graph_count=len(decoder.prefill_graphs),memory=json.loads(garnet.memory_stats_json()))
  records.append(record);save(False)
  print('mistral_profile_group',json.dumps(dict(group=record['group'],exact=[row['exact_native_control'] for row in rows],tokens=[len(row['generated_ids']) for row in rows])),flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 save(len(records)==(len(cases)+batch-1)//batch)
 print('mistral_profile_closed',json.dumps(closed),flush=True)
assert len(records)==(len(cases)+batch-1)//batch,'incomplete profile collection'
