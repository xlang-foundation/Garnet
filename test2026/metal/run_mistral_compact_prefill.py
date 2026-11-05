# SPDX-License-Identifier: Apache-2.0
"""Native compact owner parity against a separate B4 cache/control session.

XLang args: MODEL LIB SOURCE FIXTURE NEW_OUTPUT. Real weights, full cache hashes,
physical last-hidden routing, fallback and epoch/lifetime guards. Timings are
diagnostic only: two sessions share scratch and run sequentially, not HTTP.
"""
import os,sys,json,time
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
model,output=sys.argv[1],sys.argv[5]
assert not os.path.exists(output);os.mkdir(output)
with open(model+'/config.json') as f:config=json.load(f)
with open(sys.argv[4]) as f:fixture=json.load(f)
decoder=None;reference=garnet.decoder_session();records=[];negative=[];prepared=None;closed=None;completed=False
def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def save():
 with open(output+'/status.json','w') as f:json.dump(dict(completed=completed,records=records,negative=negative,prepared=prepared,cleanup=closed,backend=garnet.backend_info(),library=garnet.native_library_path(),scope=__doc__),f,indent=2)
def states(owner):return [[decoder.io.verification_f32_sha256(p) for p in owner.snapshot(i)] for i in range(32)]
def rejected(label,callback):
 failed=False
 try:callback()
 except Exception as e:failed=True;print('compact_negative',label,str(e),flush=True)
 assert failed,label;negative.append(label);save()
try:
 decoder=MistralMetalDecoder(model+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_compact_prefill(128)
 assert decoder.session.compact_prefill_family_count==14
 reference.build([layer['graph'] for layer in decoder.layers],decoder.head)
 target=decoder.session
 prepared=json.loads(garnet.memory_stats_json());print('compact_prepared',json.dumps(prepared),flush=True)
 assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
 leases=[];refleases=[]
 for row in range(4):
  lease=target.claim_request_row(row+1);other=reference.claim_request_row(row+1)
  assert lease[0]==other[0]==row;leases.append(lease[1]);refleases.append(other[1])
 offsets=[0,0,0,0]
 prompt=next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids']
 def run(label,chunk,counts):
  graphs=decoder._resident_prefill_graphs(chunk)
  ids=[[prompt[(offsets[row]+t+17*row)%len(prompt)] if t<counts[row] else 0 for row in range(4)] for t in range(chunk)]
  hidden=decoder.io.stack_tokens([decoder.io.embedding(part) for part in ids])
  positions=val([offsets[row]+t if t<counts[row] else 0 for row in range(4) for t in range(chunk) for axis in range(3)],[4,chunk,3])
  slots=val([offsets[row]+t if t<counts[row] else 0 for row in range(4) for t in range(chunk)],[4,chunk])
  active=val([int(t<counts[row]) for row in range(4) for t in range(chunk)],[4,chunk])
  began=time.monotonic();target.run_leased(graphs,hidden,positions,slots,active,-1,leases);elapsed=time.monotonic()-began
  rows=target.compact_prefill_last_rows;first=target.compact_prefill_last_first
  live=[i for i in range(4) if counts[i]]
  expected=len(live) if len(live) in (1,2) and (len(live)==1 or live[1]==live[0]+1) and len(live)*chunk>4 else 0
  assert rows==expected and first==(live[0] if expected else 4),(label,rows,first,expected)
  layouts=target.layout_vector_dispatches
  reference.run_leased(graphs,hidden,positions,slots,active,-1,refleases)
  actual=states(target);wanted=states(reference)
  assert actual==wanted,('full physical KV differs',label)
  actual_hidden=decoder.io.verification_f32_sha256(target.hidden());wanted_hidden=decoder.io.verification_f32_sha256(reference.hidden())
  assert actual_hidden==wanted_hidden,('full physical hidden differs',label)
  epochs=[leases[i] if counts[i] else 0 for i in range(4)];other=[refleases[i] if counts[i] else 0 for i in range(4)]
  predictions=target.predict_rows(epochs);expected_predictions=reference.predict_rows(other)
  assert predictions==expected_predictions,('physical row predictions differ',label)
  for row in range(4):offsets[row]+=counts[row]
  records.append(dict(label=label,chunk=chunk,counts=counts,compact_rows=rows,compact_first=first,layout_vector_dispatches=layouts,
   full_KV_sha256=actual,reference_KV_sha256=wanted,hidden_sha256=actual_hidden,reference_hidden_sha256=wanted_hidden,predictions=predictions,
   reference_predictions=expected_predictions,diagnostic_wall_seconds=elapsed,memory=json.loads(garnet.memory_stats_json())))
  save();print('compact_case',label,'PASS',rows,first,flush=True)
  return graphs,hidden,positions,slots,active
 last=run('owner0',8,[8,0,0,0])
 run('owner3',8,[0,0,0,8])
 run('owners12',8,[0,8,8,0])
 run('gapped02',8,[8,0,8,0])
 run('all4',8,[8,8,8,8])
 run('small-fallback',2,[0,0,0,2])
 run('tail',16,[0,3,0,0])
 last=run('owner3-large',128,[0,0,0,128])
 run('owners12-large',128,[0,128,128,0])
 last=run('owners23-large',128,[0,0,128,128])
 graphs,hidden,positions,slots,active=last
 stale=list(leases);stale[3]=0
 rejected('missing-active-epoch',lambda:target.run_leased(graphs,hidden,positions,slots,active,-1,stale))
 rejected('late-family-preparation',lambda:decoder.prepare_compact_prefill(128))
 # Release A, retain all other owners, then reassign A's physical row.
 old=leases[0];assert target.release_request_row(0,old) and reference.release_request_row(0,refleases[0])
 lease=target.claim_request_row(99);other=reference.claim_request_row(99);assert lease[0]==other[0]==0
 leases[0]=lease[1];refleases[0]=other[1];offsets[0]=0
 stale=list(leases);stale[0]=old
 rejected('stale-reclaimed-epoch',lambda:target.run_leased(graphs,hidden,positions,slots,active,-1,stale))
 run('reclaimed-owner0',128,[128,0,0,0])
 target.reset();reference.reset();leases=[];refleases=[];offsets=[0,0,0,0]
 for row in range(4):
  a=target.claim_request_row(row+201);b=reference.claim_request_row(row+201);assert a[0]==b[0]==row;leases.append(a[1]);refleases.append(b[1])
 run('reset-owner3',128,[0,0,0,128])
 completed=True
finally:
 reference.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[key]==0,(key,closed[key])
 save();print('compact_closed',completed,json.dumps(closed),flush=True)
assert completed
