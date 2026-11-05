# SPDX-License-Identifier: Apache-2.0
"""Actual Mistral C++ token-prefix API, B4 row reuse and survivor isolation.

XLang args: LIB SOURCE FIXTURE CODING_CONTROL OUTPUT.
Stores checkpoints64/84/115 in a96MiB host budget. Tests exact prompt, appended
generated history and a changed suffix after the stored64 boundary. Appended
and changed prompts also run fresh, independently of cache lookup. This is a
direct runtime API study; it does not admit cached HTTP chunk scheduling,
arbitrary longest-common-prefix truncation, code quality or speed.
"""
import sys,os,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder

with open(sys.argv[3]) as f:fixture=json.load(f)
cases=[c for c in fixture['cases'] if c['kind']=='coding']
controls={}
for case in cases[:5]:
 with open(sys.argv[4]+'/'+case['id']+'/request-0.result.json') as f:controls[case['id']]=json.load(f)
 assert controls[case['id']]['prompt_ids']==case['prompt_ids']
output=sys.argv[5];assert not os.path.exists(output)
with open(fixture['model']+'/config.json') as f:config=json.load(f)
assert 'native_hybrid_token_prefix' in garnet.backend_info()
decoder=None;records=[];frames=[];captures=[];rejections=[];closed=None;prepared=None;finished=False
epochs=[0]*4;positions=[0]*4;generated=[[] for _ in range(4)];first=[-1]*4;leases=[]
identity=['mistral7b-a4b8f870-fp16g64','mps-b4-c4096-chunk128-prefix96MiB',
          'mistral-v03-native-tokenizer','text_zero_based_3axis_v1','local-prefix-study','text']
backend=garnet.backend_info();library=garnet.native_library_path()

def f32(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def save():
 with open(output,'w') as f:
  json.dump(dict(completed=finished,backend=backend,library=library,fixture=sys.argv[3],controls=sys.argv[4],
    profile=dict(batch=4,capacity=4096,maximum_chunk=128,host_cache_bytes=96*1024*1024,entries=3,identity=identity),
    prepared=prepared,records=records,frames=frames,captures=captures,rejections=rejections,leases=leases,
    survivor_rows=[dict(id=cases[r]['id'],row=r,generated_ids=generated[r],
                        expected_ids=controls[cases[r]['id']]['generated_ids'][:len(generated[r])]) for r in range(1,4)],
    cleanup=closed,scope=__doc__),f,indent=2)
def claim(request):
 lease=decoder.session.claim_request_row(request);row=int(lease[0]);epoch=int(lease[1])
 epochs[row]=epoch;positions[row]=0;generated[row]=[]
 leases.append(dict(request=request,row=row,epoch=epoch));return row
def replace(request):
 assert decoder.session.release_request_row(0,epochs[0]);epochs[0]=0
 assert claim(request)==0
def predict(row):
 mask=[epochs[i] if i==row else 0 for i in range(4)]
 return decoder.session.predict_rows(mask)[row]
def prefill(rows):
 maximum=max(len(ids) for ids in rows)
 for offset in range(0,maximum,64):
  count=max(min(64,max(0,len(ids)-offset)) for ids in rows)
  chunk=max(2,1 << (count-1).bit_length())
  valid=[min(chunk,max(0,len(ids)-offset)) for ids in rows]
  ids=[rows[row][offset+t] if t<valid[row] else 0 for row in range(4) for t in range(chunk)]
  slots=[positions[row]+t if t<valid[row] else 0 for row in range(4) for t in range(chunk)]
  active=[int(t<valid[row]) for row in range(4) for t in range(chunk)]
  values=[decoder.io.embedding([ids[row*chunk+t] for row in range(4)]) for t in range(chunk)]
  decoder.session.run_tokens(decoder._resident_prefill_graphs(chunk),decoder.io.stack_tokens(values),
    f32([slot for slot in slots for axis in range(3)],[4,chunk,3]),f32(slots,[4,chunk]),
    f32(active,[4,chunk]),-1,epochs,ids)
  assert decoder.session.kv_inplace_updates==64
  before=list(positions);positions[:]=[positions[row]+valid[row] for row in range(4)]
  frames.append(dict(stage='prefill',chunk=chunk,valid=valid,before=before,after=list(positions),epochs=list(epochs)))
def generate(label,prompt,budget,expected=None):
 begin=time.monotonic();start=[len(ids) for ids in generated];generated[0]=[]
 first[0]=predict(0)
 for step in range(budget):
  active=[int(step>0)]+[int(bool(generated[row])) for row in range(1,4)]
  predicted=[-1]*4
  if any(active):
   ids=[generated[row][-1] if active[row] else 0 for row in range(4)]
   slots=[positions[row] if active[row] else 0 for row in range(4)]
   before=list(positions)
   predicted=decoder.session.run_tokens([layer['graph'] for layer in decoder.layers],decoder.io.embedding(ids),
      f32([slot for slot in slots for axis in range(3)],[4,3]),f32(slots,[4]),f32(active,[4]),0,epochs,ids)
   assert decoder.session.kv_inplace_updates==64
   positions[:]=[positions[row]+active[row] for row in range(4)]
   frames.append(dict(stage='decode',active=active,before=before,after=list(positions),epochs=list(epochs)))
  for row in range(4):
   value=predicted[row] if active[row] else first[row]
   generated[row].append(value)
   if row>0:assert value==controls[cases[row]['id']]['generated_ids'][len(generated[row])-1],'survivor changed'
  assert generated[0][-1]!=2,'unexpected early EOS in prefix study'
 record=dict(label=label,prompt_ids=prompt,generated_ids=list(generated[0]),expected_ids=expected,
             exact_control=None if expected is None else generated[0]==expected,
             row=0,epoch=epochs[0],consumed_tokens=positions[0],
             survivor_counts_before=start[1:],survivor_counts_after=[len(ids) for ids in generated[1:]],
             cache=json.loads(decoder.session.token_prefix_stats_json()),seconds=time.monotonic()-begin)
 records.append(record);save();print('mistral_prefix_case',label,'exact',record['exact_control'],flush=True)
 return list(generated[0])
def capture(boundary):
 assert positions[0]==boundary and decoder.session.cache_token_prefix(0,epochs[0])
 captures.append(dict(boundary=boundary,row=0,epoch=epochs[0],cache=json.loads(decoder.session.token_prefix_stats_json())))
def restore(prompt,wanted):
 count=decoder.session.restore_token_prefix(0,epochs[0],prompt);assert count==wanted,(count,wanted)
 positions[0]=count
 return count
def reject_stale(fn,label):
 failed=False
 try:fn()
 except Exception:failed=True
 assert failed,label;rejections.append(label)

try:
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,4,4096,resident=True)
 decoder.prepare_prefill(128)
 assert decoder.session.configure_token_prefix(identity,3,96*1024*1024)
 prepared=json.loads(garnet.memory_stats_json())
 for request in range(1,5):assert claim(request)==request-1
 original=cases[0]['prompt_ids'];gold=controls[cases[0]['id']]['generated_ids']
 assert len(original)==84
 prefill([c['prompt_ids'][:64] for c in cases[:4]]);capture(64)
 prefill([c['prompt_ids'][64:] for c in cases[:4]]);capture(84)
 predictions=decoder.session.predict_rows(epochs)
 for row in range(4):first[row]=predictions[row]
 generate('A-initial',original,32,gold[:32]);capture(115)
 stale=epochs[0];replace(5)
 reject_stale(lambda:decoder.session.restore_token_prefix(0,stale,original),'stale-restore')
 reject_stale(lambda:decoder.session.cache_token_prefix(0,stale),'stale-capture')
 replacement=cases[4]['prompt_ids'];restore(replacement,0)
 prefill([replacement,[],[],[]]);generate('E-replacement',replacement,16,controls[cases[4]['id']]['generated_ids'][:16])
 replace(6);restore(original,84);generate('A-exact-return',original,32,gold[:32])
 appended=original+gold[:32]
 replace(7);prefill([appended,[],[],[]]);append_control=generate('append-fresh',appended,32)
 replace(8);restore(appended,115);prefill([appended[115:],[],[],[]]);generate('append-restored',appended,32,append_control)
 changed=list(original);changed[70]=(changed[70]+17)%32768
 replace(9);prefill([changed,[],[],[]]);changed_control=generate('changed-fresh',changed,16)
 replace(10);restore(changed,64);prefill([changed[64:],[],[],[]]);generate('changed-restored',changed,16,changed_control)
 assert all(record['exact_control'] is not False for record in records),'cached/frozen control token divergence'
 finished=True
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 assert closed['metal_allocated_bytes']<=4*1024*1024
 save()
assert finished
