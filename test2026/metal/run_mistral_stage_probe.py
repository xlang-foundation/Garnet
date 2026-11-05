# SPDX-License-Identifier: Apache-2.0
"""Native stage diagnosis with exact control tokens; not an HTTP speed result.

XLang args: LIB SOURCE FIXTURE CONTROL_ROOT OUTPUT CASE_ID BATCH CHUNK STEPS.
Three repetitions share one prepared process. Rows repeat the same real prompt;
they are not independent coding samples. GPU time overlaps CPU wait time.
"""
import sys,json,os,time,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as stream:fixture=json.load(stream)
case=[row for row in fixture['cases'] if row['id']==sys.argv[6]][0]
with open(sys.argv[4]+'/'+case['id']+'/request-0.result.json') as stream:control=json.load(stream)
output=sys.argv[5];batch=int(sys.argv[7]);chunk=int(sys.argv[8]);steps=int(sys.argv[9])
assert not os.path.exists(output) and batch in (1,2,4) and 2<=steps<=len(control['generated_ids'])
assert batch*chunk<=512 and len(case['prompt_ids'])+steps<=4096
with open(fixture['model']+'/config.json') as stream:config=json.load(stream)
tokenizer=garnet.tokenizer();tokenizer.open(fixture['model'])
assert tokenizer.encode(case['rendered'])==case['prompt_ids']==control['prompt_ids']
decoder=None;records=[];closed=None;prepared=None

def snapshot():
 return dict(prepare_ms=decoder.session.prepare_ms,encode_ms=decoder.session.encode_ms,
   gpu_ms=decoder.session.gpu_ms,wait_ms=decoder.session.wait_ms,
   native_wall_ms=decoder.session.last_inference_ms)

def save(completed):
 with open(output,'w') as stream:
  json.dump(dict(completed=completed,backend=garnet.backend_info(),loaded_native_library=garnet.native_library_path(),
    fixture=sys.argv[3],control=sys.argv[4],case_id=case['id'],batch=batch,chunk=chunk,steps=steps,
    prepared=prepared,records=records,cleanup=closed,
    scope='Native stage diagnosis, three repetitions in one process with identical rows; no HTTP/comparator speed, unique coding sample or peak-memory claim; GPU time overlaps wait time'),stream,indent=2)

try:
 began=time.monotonic()
 decoder=MistralMetalDecoder(fixture['model']+'/model.safetensors',config,batch,4096,resident=True)
 decoder.prepare_prefill(chunk);graph_count=len(decoder.prefill_graphs)
 prepared=dict(seconds=time.monotonic()-began,graphs=graph_count,memory=json.loads(garnet.memory_stats_json()))
 for repetition in range(3):
  decoder.reset();began=time.monotonic()
  embeddings=[decoder.io.embedding([token]*batch) for token in case['prompt_ids']]
  embedding_seconds=time.monotonic()-began
  prediction=decoder.prefill_sequence(embeddings,[[[i]*3 for i in range(len(embeddings))] for row in range(batch)])
  prefill=dict(wall_seconds=time.monotonic()-began,embedding_seconds=embedding_seconds,
     native_profile=dict(decoder.prefill_profile),physical_chunks=list(decoder.prefill_chunk_plan))
  generated=[[] for row in range(batch)];samples=[]
  for step in range(steps):
   for row in range(batch):generated[row].append(prediction[row])
   assert all(prediction[row]==control['generated_ids'][step] for row in range(batch)),'changed native control prediction'
   if step+1<steps:
    began=time.monotonic();prediction=decoder.step(prediction)
    elapsed=(time.monotonic()-began)*1000
    sample=snapshot();sample['step_wall_ms']=elapsed
    assert all(math.isfinite(value) and value>=0 for value in sample.values())
    samples.append(sample)
  assert decoder.prefill_compile_ms==0 and len(decoder.prefill_graphs)==graph_count
  assert decoder.slots==[len(case['prompt_ids'])+steps-1]*batch
  records.append(dict(repetition=repetition,generated_rows=generated,prefill=prefill,
     decode_samples=samples,memory=json.loads(garnet.memory_stats_json())))
  save(False);print('mistral_stage_probe',repetition,'tokens',steps,'decode_frames',len(samples),flush=True)
finally:
 if decoder is not None:decoder.close()
 garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  assert closed[key]==0,(key,closed[key])
 assert closed['metal_allocated_bytes']<=4*1024*1024,'unchanged close allocation guard'
 save(len(records)==3)
assert len(records)==3
