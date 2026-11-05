# SPDX-License-Identifier: Apache-2.0
"""Actual-weight adaptive native scheduling, dynamic refill and frozen token IDs.

XLang args: LIB SOURCE FROZEN_CASES NEW_OUTPUT. Exercises four live owners,
queued E, A-return during survivor decode, then B1/T512 and B2/T256. Each
prefill checks all inactive physical KV rows unchanged. No speed claim.
"""
import sys,os,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:fixture=json.load(f)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
assert not os.path.exists(sys.argv[4]);decoder=None;policy=None;finished=False;closed=None;frames=[];records=[];jobs={};done={};placements={};coverage=set();stale=None

def save():
 with open(sys.argv[4],'w') as f:json.dump(dict(completed=finished,frames=frames,records=records,coverage=sorted(coverage),cleanup=closed,library=garnet.native_library_path(),scope=__doc__),f,indent=2)
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def row_hashes():
 result=[[] for _ in range(4)]
 for layer in range(32):
  for state in decoder.session.snapshot(layer):
   for row in range(4):result[row].append(decoder.io.verification_f32_sha256(decoder.io.rows(state,row,1)))
 return result
def submit(label,case,budget):
 assert len(case['generated_ids'])>=budget
 identity=policy.submit(len(case['prompt_ids']),budget,0);assert identity>0
 jobs[identity]=dict(label=label,prompt=case['prompt_ids'],expected=case['generated_ids'][:budget],generated=[])
 return identity
def run_until_empty():
 global stale
 while jobs:
  tick=json.loads(policy.next_json());assert tick['batches'] and tick['packed']['token_count']<=512
  decode_count=sum(len(p['requests']) for p in tick['batches'] if p['stage']=='decode')
  for plan in tick['batches']:
   stage=plan['stage'];chunk=plan.get('prefill_chunk',1);requests={r['row']:r for r in plan['requests']}
   assert len(requests)==len(plan['requests'])
   if stage=='prefill':
    assert sum(r['prefill_tokens'] for r in requests.values())+decode_count<=512
    if decode_count==3 and any(jobs[r['id']]['label']=='E' and r['prompt_tokens']-r['prefill_offset']>=256 for r in requests.values()):
     assert chunk==256;coverage.add('three_decode_plus_owner256')
    if any(jobs[r['id']]['label']=='solo' for r in requests.values()) and max(r['prompt_tokens']-r['prefill_offset'] for r in requests.values())>=512:
     assert chunk==512;coverage.add('solo512')
    if len(requests)==2 and any(jobs[r['id']]['label']=='pair0' for r in requests.values()) and min(r['prompt_tokens']-r['prefill_offset'] for r in requests.values())>=256:
     assert chunk==256;coverage.add('pair256')
   for r in requests.values():
    key=(r['row'],r['epoch']);identity=r['id']
    if identity in placements:assert placements[identity]==key
    else:placements[identity]=key
   ids=[];slots=[];active=[]
   for row in range(4):
    r=requests.get(row)
    count=(r['prefill_tokens'] if stage=='prefill' else 1) if r else 0
    for t in range(chunk):
     valid=t<count;slot=r['slot']+t if valid else 0
     ids.append((jobs[r['id']]['prompt'][slot] if stage=='prefill' else r['token_id']) if valid else 0)
     slots.append(slot);active.append(int(valid))
   before=row_hashes() if stage=='prefill' else None
   if stage=='prefill':
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)])
    graphs=decoder.prefill_profile_graphs(chunk);positions=f32([s for s in slots for _ in range(3)],[4,chunk,3]);slot_tensor=f32(slots,[4,chunk]);flags=f32(active,[4,chunk])
   else:
    hidden=decoder.io.embedding(ids);graphs=[l['graph'] for l in decoder.layers];positions=f32([s for s in slots for _ in range(3)],[4,3]);slot_tensor=f32(slots,[4]);flags=f32(active,[4])
   decoder.session.run_leased(graphs,hidden,positions,slot_tensor,flags,-1,plan['epochs'])
   if stage=='prefill':
    after=row_hashes()
    for row in range(4):
     if row not in requests:assert before[row]==after[row],('inactive KV changed',row)
   final_epochs=[0]*4
   for row,r in requests.items():
    if stage=='decode' or r['prefill_offset']+r['prefill_tokens']==r['prompt_tokens']:final_epochs[row]=r['epoch']
   predicted=decoder.session.predict_rows(final_epochs) if any(final_epochs) else None
   for row,r in requests.items():
    identity=r['id'];job=jobs[identity]
    if not final_epochs[row]:assert policy.complete_prefill_chunk(identity,r['prefill_tokens'])
    else:
     token=predicted[row];assert token==job['expected'][len(job['generated'])],('frozen token mismatch',job['label'],len(job['generated']),token)
     job['generated'].append(token);stop=len(job['generated'])==len(job['expected'])
     fn=policy.complete_prefill if stage=='prefill' else policy.complete_decode
     assert fn(identity,token,stop)==(3 if stop else 2)
     if stop:
      done[job['label']]=job;records.append(dict(label=job['label'],identity=identity,row=row,epoch=r['epoch'],generated_ids=job['generated'],expected_ids=job['expected']))
      del jobs[identity]
      if job['label']=='E':
       assert all(label not in done for label in ('B','C','D'));submit('A-return',fixture['coding'][0],8);coverage.add('return_with_three_survivors')
   sample=json.loads(garnet.memory_stats_json());assert sample['metal_allocated_bytes']<=sample['metal_recommended_working_set_bytes']-256*1024*1024
   frames.append(dict(plan=plan,packed_tokens=tick['packed']['token_count'],decode_tokens=decode_count,memory=sample));save()
 assert not json.loads(policy.next_json())['batches']
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True);decoder.prepare_compact_prefill(512)
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16)
 assert policy.set_prefill_chunk(512) and not policy.set_prefill_chunk(513) and not policy.set_prefill_chunk(0)
 for i,label in enumerate(['A','B','C','D']):submit(label,fixture['coding'][i],2 if i==0 else 40)
 submit('E',fixture['long'],8);run_until_empty()
 a,e,r=[next(x for x in records if x['label']==label) for label in ('A','E','A-return')]
 assert a['row']==e['row']==r['row'] and a['epoch']<e['epoch']<r['epoch'];coverage.add('fresh_epochs_same_row')
 submit('solo',fixture['long'],8);run_until_empty()
 submit('pair0',fixture['long'],8);submit('pair1',fixture['long'],8);run_until_empty()
 assert coverage==set(['three_decode_plus_owner256','solo512','pair256','return_with_three_survivors','fresh_epochs_same_row'])
 finished=True;print('wide_owner_scheduler_PASS',len(records),len(frames),sorted(coverage),flush=True)
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[k]==0
 assert closed['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
