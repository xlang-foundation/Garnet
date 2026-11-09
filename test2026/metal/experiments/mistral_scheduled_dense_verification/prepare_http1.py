"""Prepare a persistent diagnostic B1 HTTP worker over the qualified native bridge."""
from pathlib import Path
import ast,hashlib,json
from jinja2.sandbox import SandboxedEnvironment
M=Path(__file__).resolve().parent;R=M/'mistral-scheduled-dense-http1';N=M/'mistral-scheduled-dense-verification1';B=M/'mistral-bounded-draft-long1'
assert not R.exists();assert json.loads((N/'qualification-audit2.json').read_text())['passed'];R.mkdir()
i=json.loads((N/'input1.json').read_text());i.pop('cases');tc=json.loads((Path(i['draft_model'])/'tokenizer_config.json').read_text());template=SandboxedEnvironment().from_string(tc['chat_template']);render=lambda c:template.render(messages=[dict(role='user',content=c)],add_generation_prompt=True,enable_thinking=False,tools=None)
marker='NATIVE_CONTENT_MARKER_3e6f';s=render(marker);assert s.count(marker)==1;prefix,suffix=s.split(marker)
for text in ['',' x ','\ncode\n','é 中文','<|im_start|>','```python\nprint(1)\n```']:assert render(text)==prefix+text+suffix
profile=dict(profile_id='mistral7b-b1-bounded192-native-scheduler-v1',runtime_inputs=i,helper_chat_prefix=prefix,helper_chat_suffix=suffix,active_rows=1,max_pending=16,prefill_chunk=128,prefix_cache=False,model='local-mistral-7b',scope='Diagnostic B1 native-scheduled speculation serving; C4/T4 capability remains unqualified',goal_complete=False)
(R/'profile1.json').write_text(json.dumps(profile,indent=2)+'\n');(R/'plan1.json').write_text(json.dumps(dict(scope=profile['scope'],sequence=[[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]],warm_waves=[1,2],prompt_tokens=2048,output_tokens=128,prefix_cache=False,worker_active_rows=1,queued_arrivals='Bounded HTTP routing; one admitted native owner at a time',reference_max_sequences=4,c4_throughput_not_admitted=True,matched_maximum_concurrency_profile=False,no_comparator_execution=True,no_source_model_hash_or_build_during_collection=True,goal_complete=False),indent=2)+'\n')
s=(B/'worker2.py').read_text();helpers=s[s.index('def confidence_ids'):s.index('try:\n began=time.monotonic()')]
s=(N/'worker2.py').read_text();setup=s[s.index('try:\n began=time.monotonic()'):s.index(' def prefilling')];setup=setup.replace(" print('cross_draft_ready',prepared['memory']['metal_allocated_bytes'],flush=True)",'')
methods=s[s.index(' def prefilling'):s.index(' rejections=[]')]
methods=methods.replace("assert policy.complete_prefill(identity,known,False)==2", "assert policy.complete_prefill(identity,known,known in inputs['target_stops'])==(3 if known in inputs['target_stops'] else 2)")
header='''"""Persistent diagnostic B1 native scheduled speculation HTTP worker."""
import sys,json,os,time,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.mistral.text_7b.metal_decoder import MistralMetalDecoder
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
with open(sys.argv[3]) as f:profile=json.load(f)
assert len(sys.argv)==4 and profile['active_rows']==1 and not profile['prefix_cache']
inputs=profile['runtime_inputs'];fused_enabled=False
with open(inputs['target_model']+'/config.json') as f:config=json.load(f)
with open(inputs['draft_model']+'/config.json') as f:dconfig=json.load(f)
target_tokenizer=garnet.tokenizer();target_tokenizer.open(inputs['target_model'])
draft_tokenizer=garnet.tokenizer();draft_tokenizer.open(inputs['draft_model'])
decoder=None;draft=None;policy=None;request=0;draft_lease=None;use_confidence=False;stop_threshold=None;current_confidence=None

def val(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def emit(d):print(json.dumps(d,separators=(',',':')),flush=True)
def messages():
 pending=b''
 while True:
  block=os.read(0,65536)
  if not block:
   assert not pending
   return
  pending+=block
  while b'\\n' in pending:
   line,pending=pending.split(b'\\n',1)
   if len(line)>131072:raise ValueError('Worker request frame too large')
   yield json.loads(line)
  if len(pending)>131072:raise ValueError('Worker request frame too large')
'''
body='''
 emit(dict(event='ready',backend=garnet.backend_info(),profile=profile,active_requests=1,capacity=4096,prefill_chunk=128,prefix_cache=False,multimodal=False,target_graphs=target_graphs,draft_graphs=draft_graphs,memory=json.loads(garnet.memory_stats_json())))
 for message in messages():
  key=message['id'];body=message['body'];limit=body['max_tokens'];rendered=body['rendered_prompt'];user=body['draft_user_content']
  if type(limit) is not int or not 2<=limit<=128 or not isinstance(rendered,str) or not isinstance(user,str) or len(rendered)>65536:
   emit(dict(event='error',id=key,error='Request exceeds diagnostic profile',fatal=False));continue
  ids=list(target_tokenizer.encode(rendered))
  if not ids or len(ids)+limit>4096:
   emit(dict(event='error',id=key,error='Context exceeds diagnostic profile',fatal=False));continue
  tail=draft_tokenizer.decode_ex(draft_tokenizer.encode_ex(user,False)[-192:],False)
  case=dict(prompt_ids=ids,draft_prompt=profile['helper_chat_prefix']+tail+profile['helper_chat_suffix'])
  emit(dict(event='native_admit',id=key,prompt_ids=ids,draft_prompt=case['draft_prompt'],limit=limit))
  identity,epoch,known,ticks=prefilling(case,limit);generated=[];text_ids=[];target_calls=0;accepted=0;proposed=0
  def deliver(token):
   generated.append(token)
   if token not in inputs['target_stops']:text_ids.append(token)
   text=target_tokenizer.decode_ex(text_ids,True).encode('utf8').decode('utf8','replace')
   emit(dict(event='token',id=key,token_id=token,sampled_tokens=len(generated),prompt_tokens=len(ids),cached_tokens=0,text=text))
   return text
  text=deliver(known);slot=len(ids);done=known in inputs['target_stops']
  while not done:
   tick=json.loads(policy.next_json());assert len(tick['batches'])==1;plan=tick['batches'][0];req=plan['requests'][0]
   assert plan['stage']=='decode' and len(plan['requests'])==1 and req['id']==identity and req['epoch']==epoch and req['token_id']==known and req['slot']==slot
   remaining=limit-len(generated);proposal,trace=propose(case,generated,min(3,remaining-1),'resident') if remaining>1 else ([],dict(seconds=0.,reason='clipped_tail',aligned=False))
   args=scheduled_arguments(known,proposal,slot);began=time.monotonic();native=json.loads(policy.verify_decode_json(*args));elapsed=time.monotonic()-began;q=native['requests'][0]
   assert len(native['requests'])==1 and q['id']==identity and q['epoch']==epoch and q['remaining_before']==remaining and q['known']==known
   emit(dict(event='native_dense_verification',id=key,committed_before=len(generated),slot=slot,proposal=proposal,submitted_ids=args[4],draft=trace,native_schedule=plan,native=native,target_seconds=elapsed))
   target_calls+=1;accepted+=q['consumed']-1;proposed+=len(proposal)
   for token in q['emitted']:text=deliver(token)
   known=generated[-1];slot+=q['consumed'];done=q['finished']
  if draft_lease is not None:assert draft.session.release_request_row(0,draft_lease[1]);draft_lease=None
  assert len(generated)<=limit and not decoder.session.release_request_row(0,epoch)
  emit(dict(event='completed',id=key,generated_ids=generated,prompt_ids=ids,text=text,cached_tokens=0,finish_reason='stop' if known in inputs['target_stops'] else 'length',generation_mode='native_scheduled_dense_speculation_B1',profile_id=profile['profile_id'],native_request=identity,speculative_metrics=dict(target_calls=target_calls,accepted=accepted,proposed=proposed),native_status=json.loads(policy.status_json()),prefill_ticks=ticks,memory=json.loads(garnet.memory_stats_json())))
finally:
 if policy is not None:policy.close()
 if draft is not None:draft.close()
 if decoder is not None:decoder.close()
 garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
 assert memory['metal_allocated_bytes']<=4194304
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[k]==0
 emit(dict(event='closed',memory=memory))
'''
worker=header+helpers+setup+methods+body;ast.parse(worker);(R/'worker1.py').write_text(worker)
(R/'preparation1.json').write_text(json.dumps(dict(passed=True,helper_algorithm_sha256=hashlib.sha256(helpers.encode()).hexdigest(),worker_sha256=hashlib.sha256(worker.encode()).hexdigest(),profile_no_saved_outputs=True,template_prefix_suffix_cpu_verified=True,goal_complete=False),indent=2)+'\n')
print('Prepared diagnostic B1 serving worker',R)
