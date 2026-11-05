# SPDX-License-Identifier: Apache-2.0
"""Real-weight wide owner parity against independent B1/B2 native KV owners.

XLang args: LIB SOURCE FIXTURE NEW_OUTPUT_DIRECTORY. No B4/T256 or B4/T512
GPU graph is constructed. Controls reuse exact immutable compact programs and
weights but own independent caches. Timings/memory are diagnostic, not speed.
"""
import os,sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.qwen.metal_dense.layers import T,norm
lib,source,fixture_path,output=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
assert not os.path.exists(output);os.mkdir(output)
assert garnet.native_library_path()==lib+'/libgarnet.dylib'
with open(fixture_path) as f:fixture=json.load(f)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
prompt=next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids']
decoder=None;reference=None;ref_head=None;records=[];rejections=[];closed=None;prepared=None;completed=False;leases=[];offsets=[0]*4;ref_epochs=[]
def f32(v,s):return tensor.tensor(v,shape=s,dtype=tensor.float32)
def memory():return json.loads(garnet.memory_stats_json())
def save():
 with open(output+'/status.json','w') as f:json.dump(dict(completed=completed,library=garnet.native_library_path(),backend=garnet.backend_info(),prepared=prepared,
  records=records,rejections=rejections,cleanup=closed,scope=__doc__),f,indent=2)
def hashes(owner,rows):
 result=[[] for row in range(rows)]
 for layer in range(32):
  states=owner.snapshot(layer)
  for row in range(rows):
   result[row].append([decoder.io.verification_f32_sha256(decoder.io.rows(p,row,1)) for p in states])
 return result
def claim_target():
 global leases,offsets
 leases=[];offsets=[0]*4
 for row in range(4):
  value=decoder.session.claim_request_row(row+1);assert value[0]==row;leases.append(value[1])
def reset_ref(rows):
 global ref_epochs
 reference.reset();ref_epochs=[]
 for row in range(rows):
  value=reference.claim_request_row(row+1);assert value[0]==row;ref_epochs.append(value[1])
def graph_list(rows,chunk):return [decoder.compact_prefill_graphs[str(rows)+'-'+str(chunk)+'-'+str(i)] for i in range(32)]
def inputs(chunk,counts):
 ids=[[prompt[(offsets[row]+t+17*row)%len(prompt)] if t<counts[row] else 0 for row in range(4)] for t in range(chunk)]
 hidden=decoder.io.stack_tokens([decoder.io.embedding(part) for part in ids])
 slots=[offsets[row]+t if t<counts[row] else 0 for row in range(4) for t in range(chunk)]
 active=[int(t<counts[row]) for row in range(4) for t in range(chunk)]
 return (hidden,f32([v for v in slots for axis in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),f32(active,[4,chunk]))
def ref_run(args,first,rows,chunk):
 hidden,pos,slots,active=args
 reference.run_leased(graph_list(rows,chunk),decoder.io.rows(hidden,first,rows),
  decoder.io.rows(pos,first,rows),decoder.io.rows(slots,first,rows),decoder.io.rows(active,first,rows),-1,ref_epochs)
def run(label,chunk,counts):
 live=[i for i in range(4) if counts[i]];first=live[0];rows=len(live);assert rows in (1,2) and live==list(range(first,first+rows))
 args=inputs(chunk,counts);before=hashes(decoder.session,4);start=time.monotonic()
 decoder.session.run_leased(decoder.prefill_profile_graphs(chunk),args[0],args[1],args[2],args[3],-1,leases)
 elapsed=time.monotonic()-start
 assert decoder.session.compact_prefill_last_rows==rows and decoder.session.compact_prefill_last_first==first
 assert decoder.session.layout_vector_dispatches==257 and decoder.session.kv_inplace_updates==64
 ref_run(args,first,rows,chunk);actual=hashes(decoder.session,4);control=hashes(reference,rows)
 assert actual[first:first+rows]==control,('independent wide KV differs',label)
 for row in range(4):
  if row not in live:assert actual[row]==before[row],('unrelated physical owner changed',label,row)
 full=decoder.session.hidden();selected=decoder.io.rows(full,first,rows);ref_hidden=reference.hidden()
 a,b=decoder.io.verification_f32_sha256(selected),decoder.io.verification_f32_sha256(ref_hidden)
 assert a==b,('wide hidden differs',label)
 predicted=decoder.session.predict_rows([leases[row] if counts[row] else 0 for row in range(4)]);other=reference.predict_rows(ref_epochs)
 assert predicted[first:first+rows]==other and all(predicted[row]==-1 for row in range(4) if row not in live),('wide prediction routing differs',label)
 for row in range(4):offsets[row]+=counts[row]
 sample=memory();assert sample['metal_allocated_bytes']<=sample['metal_recommended_working_set_bytes']-256*1024*1024
 records.append(dict(label=label,chunk=chunk,counts=counts,first=first,rows=rows,physical_KV_sha256=actual,reference_KV_sha256=control,
  hidden_sha256=a,reference_hidden_sha256=b,predictions=predicted,reference_predictions=other,memory=sample,diagnostic_wall_seconds=elapsed))
 save();print('wide_owner_case',label,'PASS',flush=True);return args
def rejected(label,callback):
 before=hashes(decoder.session,4);failed=False
 try:callback()
 except Exception as e:failed=True;print('wide_owner_rejected',label,str(e),flush=True)
 assert failed and before==hashes(decoder.session,4),('negative case changed private cache',label)
 rejections.append(label);save()
def make_reference(rows,chunk):
 global reference,ref_head
 symbol=tensor.input('wide_control_head',shape=[rows,4096],dtype=tensor.float32)
 @T.fusion(name='wide_control_head',boundary='required')
 def head(hidden):return norm(hidden,'model.norm',decoder.config)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=1)
 ref_head=garnet.compiled_graph();ref_head.build_shared(head(symbol),[symbol],decoder.checkpoint,decoder.head)
 reference=garnet.decoder_session();reference.build(graph_list(rows,chunk),ref_head);reset_ref(rows)
try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_compact_prefill(512)
 assert decoder.session.compact_prefill_family_count==17 and decoder.session.compact_prefill_max_chunk==512
 assert [decoder.session.chunk_attention_mps_plan_count,decoder.session.chunk_attention_mps_scratch_bytes,decoder.session.chunk_attention_mps_bucket_count]==[24,1885120512,144]
 assert len(decoder.prefill_graphs)==7*32 and len(decoder.compact_prefill_graphs)==17*32
 prepared=memory();assert prepared['metal_allocated_bytes']+2*1024*1024*1024+64*1024*1024<=prepared['metal_recommended_working_set_bytes']-256*1024*1024;save();print('wide_owner_prepared',json.dumps(prepared),flush=True)
 make_reference(1,512);claim_target()
 seed0=run('owner0-512',512,[512,0,0,0]);reset_ref(1)
 run('owner3-512',512,[0,0,0,512]);run('owner3-append256',256,[0,0,0,256])
 reset_ref(1);ref_run(seed0,0,1,512);run('owner0-tail16',16,[9,0,0,0])
 stale=list(leases);old=leases[0];assert decoder.session.release_request_row(0,old)
 new=decoder.session.claim_request_row(99);assert new[0]==0;leases[0]=new[1];offsets[0]=0
 args=inputs(512,[512,0,0,0]);reset_ref(1)
 rejected('stale-reclaimed-wide-epoch',lambda:decoder.session.run_leased(decoder.prefill_profile_graphs(512),args[0],args[1],args[2],args[3],-1,stale))
 run('reclaimed0-512',512,[512,0,0,0])
 decoder.session.reset();claim_target();reset_ref(1);run('reset-owner3-512',512,[0,0,0,512])
 reference.close();reference=None;ref_head.close();ref_head=None
 decoder.session.reset();claim_target();make_reference(2,256)
 run('owners12-256',256,[0,256,256,0]);run('owners12-ragged256',256,[0,256,200,0])
 decoder.session.reset();claim_target();reset_ref(2);run('owners23-256',256,[0,0,256,256])
 args=inputs(256,[256,0,256,0])
 rejected('gapped-wide256',lambda:decoder.session.run_leased(decoder.prefill_profile_graphs(256),args[0],args[1],args[2],args[3],-1,leases))
 args=inputs(512,[512,512,0,0])
 rejected('two-owners-exceed-global512',lambda:decoder.session.run_leased(decoder.prefill_profile_graphs(512),args[0],args[1],args[2],args[3],-1,leases))
 args=inputs(512,[512,0,0,0])
 rejected('unleased-wide512',lambda:decoder.session.run(decoder.prefill_profile_graphs(512),args[0],args[1],args[2],args[3],-1))
 rejected('late-wide-preparation',lambda:decoder.prepare_compact_prefill(512))
 completed=True
finally:
 if reference is not None:reference.close()
 if ref_head is not None:ref_head.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=memory()
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[key]==0
 assert closed['metal_allocated_bytes']<=4*1024*1024
 save();print('wide_owner_closed',completed,json.dumps(closed),flush=True)
assert completed
