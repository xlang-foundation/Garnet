# SPDX-License-Identifier: Apache-2.0
"""Native B4/C4096/T128 generation with one early finish and three survivors.

Args LIB SOURCE FROZEN_INPUT OUTPUT ENGINE. Four distinct pinned coding prompts;
row0 stops after2 generated IDs, other rows after128. Native scheduler leases
and prepared prefill families are used. Only steady three-row native decode
commands are timed; this is not HTTP/TTFT or a serving throughput admission.
"""
import json,os,sys,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];engine=sys.argv[5];assert engine in ('control','candidate') and not os.path.exists(output)
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert ('mistral_decode_reuse_three_rows' in garnet.backend_info())==(engine=='candidate')
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
decoder=None;policy=None;cleanup=None;finished=False;jobs={};records=[];samples=[];plans=[];inactive_checks=0;before=None;after=None;prepared=None
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def save():
 with open(output,'w') as f:json.dump(dict(completed=finished,engine=engine,library=garnet.native_library_path(),backend=garnet.backend_info(),records=records,samples=samples,plans=plans,inactive_before=before,inactive_after=after,inactive_row_checks=inactive_checks,prepared=prepared,cleanup=cleanup,scope=__doc__),f,indent=2)
def row_state(row):
 return [[decoder.io.verification_f32_sha256(decoder.io.rows(p,row,1)) for p in decoder.session.snapshot(i)] for i in range(32)]
try:
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_compact_prefill(128);policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16) and policy.set_prefill_chunk(128)
 preparation_compile_ms=decoder.prefill_compile_ms;decoder.prefill_compile_ms=0
 base=[l['graph'] for l in decoder.layers];graph_count=len(decoder.prefill_graphs)
 prepared=json.loads(garnet.memory_stats_json())
 prepared['prefill_preparation_compile_ms']=preparation_compile_ms
 prepared['prefill_graphs']=graph_count
 for index,case in enumerate(inputs['cases'][:4]):
  with open(case['control']) as f:control=json.load(f)
  budget=2 if index==0 else 128;identity=policy.submit(len(case['prompt_ids']),budget,0)
  assert identity>0 and len(control['generated_ids'])>=budget
  jobs[identity]=dict(case_id=case['id'],prompt=case['prompt_ids'],expected=control['generated_ids'][:budget],generated=[])
 while jobs:
  tick=json.loads(policy.next_json());assert tick['batches'] and tick['packed']['token_count']<=512
  for plan in tick['batches']:
   plans.append(plan);save()
   stage=plan['stage'];chunk=plan.get('prefill_chunk',128) if stage=='prefill' else 1;requests={r['row']:r for r in plan['requests']}
   assert len(requests)==len(plan['requests']);ids=[];slots=[];active=[]
   for row in range(4):
    r=requests.get(row);count=(r['prefill_tokens'] if stage=='prefill' else 1) if r else 0
    assert 0<=count<=chunk
    for t in range(chunk):
     valid=t<count;slot=r['slot']+t if valid else 0
     ids.append((jobs[r['id']]['prompt'][slot] if stage=='prefill' else r['token_id']) if valid else 0)
     slots.append(slot);active.append(int(valid))
   final_epochs=[0]*4
   for row,r in requests.items():
    if stage=='decode' or r['prefill_offset']+r['prefill_tokens']==r['prompt_tokens']:final_epochs[row]=r['epoch']
   if stage=='prefill':
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
    predicted=decoder.session.run_leased(decoder.prefill_profile_graphs(chunk),hidden,f32([s for s in slots for _ in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]),-1,plan['epochs'])
    predicted=decoder.session.predict_rows(final_epochs) if any(final_epochs) else None
   else:
    hidden=decoder.io.embedding(ids);position=f32([s for s in slots for _ in range(3)],[4,3]);cursor=f32(slots,[4]);flags=f32(active,[4])
    began=time.monotonic();predicted=decoder.session.run_leased(base,hidden,position,cursor,flags,0,plan['epochs']);elapsed=time.monotonic()-began
    count=len(requests);blocks=decoder.session.affine_decode_block_dispatches
    expected=161 if count==4 or (count==3 and engine=='candidate') else 0
    assert blocks==expected,(count,blocks,expected)
    samples.append(dict(active_rows=sorted(requests),seconds=elapsed,gpu_ms=decoder.session.gpu_ms,affine_blocks=blocks,epochs=plan['epochs']))
    if count==3:
     assert 0 not in requests and before is not None
     # Full cache hashes are outside timing, after the first survivor step
     # and after the run; avoid disturbing every intervening decode command.
     if inactive_checks==0:
      after=row_state(0);assert before==after;inactive_checks+=1
   for row,r in requests.items():
    job=jobs[r['id']]
    if not final_epochs[row]:assert policy.complete_prefill_chunk(r['id'],r['prefill_tokens'])
    else:
     token=predicted[row];assert token==job['expected'][len(job['generated'])],(job['case_id'],len(job['generated']),token)
     job['generated'].append(token);stop=len(job['generated'])==len(job['expected'])
     committed_state=row_state(row) if stop else None
     committed_hidden=decoder.io.verification_f32_sha256(decoder.io.rows(decoder.session.hidden(),row,1)) if stop else None
     fn=policy.complete_prefill if stage=='prefill' else policy.complete_decode
     assert fn(r['id'],token,stop)==(3 if stop else 2)
     if stop:
      # Completing a lease may clear its KV. Establish the retained inactive
      # control after release, before any survivor command consumes a new token.
      if row==0:before=row_state(0)
      records.append(dict(case_id=job['case_id'],row=row,epoch=r['epoch'],generated_ids=job['generated'],expected_ids=job['expected'],state_sha256=committed_state,hidden_sha256=committed_hidden))
      del jobs[r['id']]
   assert len(decoder.prefill_graphs)==graph_count and decoder.prefill_compile_ms==0
   save()
 after=row_state(0);assert before==after;inactive_checks+=1
 assert not json.loads(policy.next_json())['batches'] and len(records)==4 and inactive_checks==2
 finished=True;print('three_active_rows_PASS',len(samples),'commands',inactive_checks,'inactive KV checks',flush=True)
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert cleanup[key]==0
 assert cleanup['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
