"""Native prefill diagnostic: same prepared engine, alternating optional down kernel; no HTTP/decode claim."""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
with open(sys.argv[3]) as f:inputs=json.load(f)
output=sys.argv[4];order=sys.argv[5];assert order in ('forward','reverse') and not os.path.exists(output)
def val(a,shape):return tensor.tensor(a,shape=shape,dtype=tensor.float32)
decoder=None;policy=None;prepared=None;cleanup=None;waves=[];finished=False
def save(done):
 with open(output,'w') as f:json.dump(dict(completed=done,inputs=inputs,order=order,backend=garnet.backend_info(),library=garnet.native_library_path(),prepared=prepared,waves=waves,cleanup=cleanup,scope=__doc__),f,indent=2)
try:
 with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
 tokenizer=garnet.tokenizer();tokenizer.open(inputs['target_model'])
 for item in inputs['jobs']:assert tokenizer.encode(item['rendered'])==item['prompt_ids'] and item['limit']==1
 decoder=MistralMetalDecoder(inputs['target_model']+'/model.safetensors',config,4,4096);decoder.prepare_prefill(128);decoder.prepare_compact_prefill(512);decoder.prepare_verification(4);decoder.prefill_compile_ms=0
 assert 'native_mistral_down_tensor512_optional_v1' in garnet.backend_info()
 assert decoder.session.configure_token_tracking(['d7182d886f93a9efa73d6d556ee33ffe839a4e09e15965eeef46f295ab5b503d',decoder.io.checkpoint_format,inputs['target_tokenizer_sha256'],'text_zero_based_3axis_v1','scheduled-B4-C4096-prefill-up512-verify4-fp32-v1','text'])
 policy=garnet.continuous_scheduler();assert policy.initialize(decoder.session,16) and policy.set_prefill_chunk(512)
 prepared=dict(memory=json.loads(garnet.memory_stats_json()),target_graphs=len(decoder.prefill_graphs),compact_families=decoder.session.compact_prefill_family_count)
 assert prepared['memory']['metal_allocated_bytes']<=prepared['memory']['metal_recommended_working_set_bytes']-536870912
 print('native_prefill_measure_ready',order,flush=True)
 pattern=[False,True,True,False]
 if order=='reverse':pattern=[not x for x in pattern]
 for wave in range(12):
  enabled=pattern[wave%4];assert decoder.session.prefill_set_tensor_down(enabled)
  jobs={};all_jobs={};trace=[];began=time.monotonic()
  for item in inputs['jobs']:
   identity=policy.submit_tokens(item['prompt_ids'],1,0);assert identity>0
   job=dict(label=item['label'],prompt_ids=item['prompt_ids'],prefilled=0,generated=[],native_request=identity);jobs[identity]=job;all_jobs[identity]=job
  while jobs:
   tick=json.loads(policy.next_json());assert tick['batches']
   for plan in tick['batches']:
    assert plan['stage']=='prefill';width=plan.get('prefill_chunk',128);byrow={q['row']:q for q in plan['requests']};ids=[];slots=[];active=[];final=[0]*4
    for row in range(4):
     q=byrow.get(row)
     if q is not None and q['prefill_offset']+q['prefill_tokens']==len(jobs[q['id']]['prompt_ids']):final[row]=q['epoch']
     for t in range(width):
      on=q is not None and t<q['prefill_tokens'];position=q['prefill_offset']+t if on else 0
      ids.append(jobs[q['id']]['prompt_ids'][position] if on else 0);slots.append(float(position));active.append(float(on))
    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*width+t] for row in range(4)]) for t in range(width)])
    decoder.session.run_tokens(decoder.prefill_profile_graphs(width),hidden,val([s for s in slots for a in range(3)],[4,width,3]),val(slots,[4,width]),val(active,[4,width]),-1,plan['epochs'],ids)
    dispatches=decoder.session.packed_tensor_down_dispatches
    predicted=decoder.session.predict_rows(final) if any(final) else None
    for q in plan['requests']:
     j=jobs[q['id']];assert j['prefilled']==q['prefill_offset'];j['prefilled']+=q['prefill_tokens'];j.update(row=q['row'],epoch=q['epoch'])
     if final[q['row']]:
      j['generated']=[predicted[q['row']]];j['first_prediction_seconds']=time.monotonic()-began;assert policy.complete_prefill(q['id'],j['generated'][0],True)==3;del jobs[q['id']]
     else:assert policy.complete_prefill_chunk(q['id'],q['prefill_tokens'])
    trace.append(dict(width=width,active_tokens=int(sum(active)),rows=[q['row'] for q in plan['requests']],dispatches=dispatches))
  ended=time.monotonic()
  status=json.loads(policy.status_json());assert all(status[k]==0 for k in ['active','queued','tracked','inflight'])
  assert decoder.prefill_compile_ms==0
  waves.append(dict(wave=wave,warmup=wave<4,enabled=enabled,began=began,ended=ended,elapsed_seconds=ended-began,jobs=list(all_jobs.values()),route_trace=trace))
  save(False);print('native_prefill_wave',order,wave,enabled,ended-began,flush=True)
 finished=True
finally:
 if policy is not None:policy.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();cleanup=json.loads(garnet.memory_stats_json());assert all(cleanup[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']) and cleanup['metal_allocated_bytes']<=4194304
 save(finished)
