"""Native scheduler owns verified block budgets, row reuse and completion."""
import json,os,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
from xModel.qwen38.vl_27b.metal_mtp_study import MetalMTPStudy
with open(sys.argv[1]) as stream:f=json.load(stream)
with open(f['config']) as stream:config=json.load(stream)
with open(sys.argv[5]) as stream:tasks=json.load(stream)
study=None;policy=None;records=[];frames=[];prefills=[];refreshes=0;combined=0
try:
 study=MetalMTPStudy(f['checkpoint'],config,sys.argv[4],4096,batch=2)
 policy=garnet.continuous_scheduler();policy.initialize(study.session,16);policy.set_prefill_chunk(128)
 tokenizer=garnet.tokenizer();tokenizer.open(os.path.dirname(f['config']));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 prepared=json.loads(garnet.memory_stats_json());print('scheduled_mtp_prepared',json.dumps(prepared),flush=True)
 profile_packs=[program.graph.mtp_deferred_pack_slots for program in study.programs]
 assert len(profile_packs)==2 and all(count>0 for count in profile_packs)
 with open(f['output']+'.inventory.json','w') as stream:stream.write(study.inventory())
 os.mkdir(tasks['responses'])
 def save(passed):
  with open(f['output'],'w') as stream:json.dump(dict(passed=passed,tasks=records,frames=frames,prefills=prefills,prepared_memory=prepared,deferred_auxiliary_pack_slots=profile_packs,combined_frames=combined,prediction_refreshes=refreshes,scope='B2 native scheduled coding/full IDs and real2K prompts; no HTTP or repeated performance admission'),stream,indent=2)
 def finish(identity,job):
  generated=job['generated'];plan=job['plan'];equal=generated==plan['expected']
  text=tokenizer.decode(generated[:-1] if generated[-1] in stops else generated).encode('utf8').decode('utf8','replace')
  row=dict(id=plan['id'],prompt=job['prompt'],generated_ids=generated,expected_ids=plan['expected'],full_token_match=equal,transactions=job['transactions'],coding=plan['coding'],finish_reason='stop' if generated[-1] in stops else 'length')
  records.append(row)
  if plan['coding']:
   folder=tasks['responses']+'/'+plan['id'];os.mkdir(folder)
   with open(folder+'/request-0.result.json','w') as stream:json.dump(dict(status='completed',finish_reason=row['finish_reason'],text=text,generated_ids=generated),stream,indent=2)
  save(False);print('scheduled_mtp_task',json.dumps(dict(id=plan['id'],tokens=len(generated),full_token_match=equal)),flush=True)
  assert equal,'Changed frozen output: '+plan['id']
 for is_coding in (True,False):
  jobs={}
  for plan in tasks['plans']:
   if plan['coding']!=is_coding:continue
   identity=policy.submit(len(plan['prompt']),plan['max_output'],0);assert identity>0
   jobs[identity]=dict(prompt=plan['prompt'],plan=plan,generated=[],transactions=[])
  while jobs:
   tick=json.loads(policy.next_json());assert tick['batches'],'Native scheduler stalled'
   for batch in tick['batches']:
    if batch['stage']=='prefill':
     predictions=study.prefill_scheduled(batch['requests'],jobs)
     for request in batch['requests']:
      identity=request['id'];job=jobs[identity];final=request['prefill_offset']+request['prefill_tokens']==len(job['prompt'])
      prefills.append(dict(id=identity,row=request['row'],epoch=request['epoch'],offset=request['prefill_offset'],tokens=request['prefill_tokens'],final=final))
      if final:
       token=predictions[request['row']];stage=policy.complete_prefill(identity,token,token in stops);assert stage in (2,3)
       job['generated'].append(token)
       if stage==3:finish(identity,job);jobs.pop(identity)
      else:assert policy.complete_prefill_chunk(identity,request['prefill_tokens'])
    else:
     result=json.loads(policy.mtp_generate_json(stops));result['auxiliary']=json.loads(study.session.mtp_stats_json());result['hidden_pool_bindings']=study.session.prefill_hidden_pool_bindings if 'native_prefill_hidden_pool' in garnet.backend_info() else 0;frames.append(result);combined+=int(len(result['requests'])==2);refreshes+=int(result['generation']['target_prediction_refreshed'])
     if len(frames)%32==0:print('scheduled_mtp_progress',json.dumps(dict(frames=len(frames),completed=len(records),active_ids=[item['id'] for item in result['requests']])),flush=True)
     if 'native_mtp_kv_inplace' in garnet.backend_info():assert result['auxiliary']['kv_inplace_updates']==2
     assert result['generation']['recursive_draft_calls']==0,'B2 should not run recursive auxiliary drafts'
     assert result['generation']['correction_profile_chunk']==2
     if 'native_prefill_hidden_pool' in garnet.backend_info():assert study.session.prefill_hidden_pool_bindings==64
     for item in result['requests']:
      job=jobs[item['id']];assert item['known']==job['generated'][-1]
      assert item['remaining_before']==job['plan']['max_output']-len(job['generated'])
      assert item['consumed']==len(item['emitted']) and 1<=item['consumed']<=2
      job['generated']+=item['emitted'];job['transactions'].append(item)
      assert item['finished']==(item['eos'] or len(job['generated'])==job['plan']['max_output'])
      if item['finished']:finish(item['id'],job);jobs.pop(item['id'])
 assert combined>0 and refreshes>0
 assert json.loads(policy.status_json())['tracked']==0
 save(True)
finally:
 if policy is not None:policy.close()
 if study is not None:study.close()
 garnet.shutdown();memory=json.loads(garnet.memory_stats_json())
 for name in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert memory[name]==0,(name,memory[name])
 print('scheduled_mtp_closed',json.dumps(memory),flush=True)
