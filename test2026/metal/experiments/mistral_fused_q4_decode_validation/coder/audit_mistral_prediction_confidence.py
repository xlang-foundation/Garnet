#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently re-tokenize native learned-draft traces and audit target commits.

Uses pinned Hugging Face tokenizer JSONs as CPU-only reference. This audits
complete native confidence-capture/stopping stage evidence; it cannot establish HTTP speed or coding quality.
"""
import argparse,copy,hashlib,json,math
from pathlib import Path
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment

def require(value,message):
 if not value:raise ValueError(message)
def duration(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def validate(data,reference,draft_family='qwen3'):
 require(data['completed'] is True,'incomplete experiment')
 inputs=data['inputs'];require((inputs['capacity'],inputs['output_limit'],inputs['draft_token_limit'])==(4096,1024,8),'changed profile')
 occupancy=data.get('projection_occupancy')
 require(occupancy in (None,'occupancy-control','occupancy-candidate'),'unknown projection experiment')
 require(inputs['confidence_threshold']==1.0,'changed frozen threshold')
 require(len(inputs['cases'])==8 and len({c['id'] for c in inputs['cases']})==8,'missing input case')
 require(draft_family in ('qwen3','qwen2'),'unrecognized draft family')
 if draft_family=='qwen3':
  require(inputs['draft_revision']=='c1899de289a04d12100db370d81485cdf75e47ca','unrecognized draft checkpoint')
  draft_layers,draft_heads,draft_dim,draft_width=28,8,128,1024
 else:
  require(inputs['draft_revision']=='6b16732e5af5cd9bd600186ad59fa618867ef7a4' and inputs.get('draft_source_repo')=='mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit','unrecognized coder draft checkpoint')
  require(inputs['draft_checkpoint_format']=='mlx_qwen2_affine_int4_g64_fp16','wrong coder draft representation')
  draft_layers,draft_heads,draft_dim,draft_width=24,2,64,896
 draft_config=json.loads((Path(inputs['draft_model'])/'config.json').read_text())
 require(draft_config['model_type']==draft_family and draft_config['num_hidden_layers']==draft_layers and draft_config['num_key_value_heads']==draft_heads and draft_config['hidden_size']==draft_width and draft_config.get('head_dim',draft_width//draft_config['num_attention_heads'])==draft_dim,'changed draft geometry')
 require('native_greedy_dense_verification' in data['backend'] and 'native_hybrid_token_prefix' in data['backend'],'wrong runtime')
 require('prediction_confidence_top2_v1' in data['backend'],'missing confidence capability')
 require('verification_value_route_control' in data['backend'] and 'mistral_verification_value_vector4' in data['backend'],'missing vector capability')
 require('verification_layout_route_control' in data['backend'] and 'mistral_verification_layout_b1t4_c4096_h32_kv8_d128' in data['backend'],'wrong compiled layout/control capability')
 if occupancy:require(('mistral_decode_reuse_three_rows' in data['backend'])==(occupancy=='occupancy-candidate'),'wrong three-row engine')
 record_width=1 if occupancy else 3
 require(len(data['records'])==8*record_width,'missing/extra run')
 require(data['prepared']['target_graphs']==224 and data['prepared']['draft_graphs']==draft_layers*4,'changed prepared family')
 for memory in [data['prepared']['memory']]+[r['memory'] for r in data['records']]:
  require(0<=memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-512*1024*1024,'memory budget')
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  require(data['cleanup'][key]==0,'live owner after close')
 require(0<=data['cleanup']['metal_allocated_bytes']<=4*1024*1024,'close allocation')
 target=Tokenizer.from_file(inputs['target_model']+'/tokenizer.json');draft=Tokenizer.from_file(inputs['draft_model']+'/tokenizer.json')
 summaries=[];resident_calls=0
 witnesses={x['case_id']:x for x in reference['records'] if x['mode']=='control'}
 require(len(witnesses)==8,'missing frozen reference')
 for index,case in enumerate(inputs['cases']):
  pair=data['records'][record_width*index:record_width*index+record_width];modes=['base'] if occupancy else (['base','scores','stop'] if index%2==0 else ['stop','scores','base'])
  require([r['mode'] for r in pair]==modes and all(r['case_id']==case['id'] for r in pair),'case order')
  raw=json.loads(Path(case['control']).read_text());require(raw['prompt_ids']==case['prompt_ids'],'control prompt')
  for row in pair:
   ids=row['generated_ids'];require(ids==raw['generated_ids'] and ids[-1] in inputs['target_stops'] and len(ids)<=1024,'target IDs differ')
   require(row['prompt_ids']==case['prompt_ids'] and row['slots']==len(case['prompt_ids'])+len(ids)-1,'target slot')
   require(duration(row['prefill_seconds']) and duration(row['decode_seconds']) and row['decode_seconds']>0,'invalid timing')
   require(abs(row['decode_tokens_per_second']-(len(ids)-1)/row['decode_seconds'])<=1e-9,'token rate arithmetic')
   require(len(row['state_sha256'])==32 and all(len(p)==2 and all(digest(h) for h in p) for p in row['state_sha256']) and digest(row['hidden_sha256']),'invalid state hash')
   require(row['state_sha256']==witnesses[case['id']]['state_sha256'] and row['hidden_sha256']==witnesses[case['id']]['hidden_sha256'],'frozen target state differs')
   committed=1;hits=0;accepted=0;proposed=0;resident_history=[]
   for step in row['rounds']:
    require(step['committed_before']==committed and step['slot']==len(case['prompt_ids'])+committed-1,'noncausal cursor')
    proposal=step['proposal'];trace=step['draft'];remaining=1024-committed
    require(0<=len(proposal)<=min(3,remaining-1),'proposal budget')
    require(duration(trace['seconds']) and duration(step['target_seconds']),'invalid component time')
    if trace['reason'] not in ('clipped_tail','noncanonical_prefix','draft_capacity'):
     prefix=target.decode(ids[:committed],skip_special_tokens=False)
     require(trace['roundtrip'] and target.encode(prefix,add_special_tokens=False).ids==ids[:committed],'target prefix identity')
     prompt=draft.encode(case['draft_prompt']+prefix,add_special_tokens=False).ids
     require(prompt==trace['prompt_ids'],'draft conditioned on wrong prefix')
     require(0<=trace['cached_tokens']<=len(prompt) and trace['missing_tokens']==len(prompt)-trace['cached_tokens'],'prefix restore cursor')
     hits+=trace['cached_tokens']>0
     generated=trace['generated_ids'];require(0<=len(generated)<=8 and not any(t in inputs['draft_stops'] for t in generated),'draft token budget/stop')
     text=draft.decode(generated,skip_special_tokens=False);require(text==trace['draft_text'],'draft text')
     translated=target.encode(prefix+text,add_special_tokens=False).ids if generated else [];require(translated==trace['translated_ids'],'translated IDs')
     aligned=translated[:committed]==ids[:committed];require(trace['aligned']==aligned,'changed alignment')
     wanted=translated[committed:committed+min(3,remaining-1)] if aligned else []
     require(proposal==wanted,'changed cross-tokenizer proposal')
     confidence=trace['confidences'];is_scored=row['mode']!='base';threshold=1.0 if row['mode']=='stop' else None
     require(trace['threshold']==threshold,'changed per-round threshold')
     require(len(confidence)==(len(generated) if is_scored else 0),'missing confidence captures')
     for token,values in zip(generated,confidence):
      require(len(values)==4 and type(values[0]) is int and values[0]==token,'confidence token')
      require(all(duration(v) or (isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)) for v in values[1:]),'nonfinite confidence')
      require(values[1]>=values[2] and values[3]==values[1]-values[2],'confidence margin')
     terminal=trace['terminal_prediction'];terminal_confidence=trace['terminal_confidence']
     if is_scored:
      require(len(terminal_confidence)==4 and terminal_confidence[0]==terminal,'terminal confidence')
      require(all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in terminal_confidence[1:]) and terminal_confidence[3]==terminal_confidence[1]-terminal_confidence[2]>=0,'terminal margin')
     else:require(terminal_confidence is None,'baseline captured scores')
     stop=None
     for i in range(len(generated)):
      partial=target.encode(prefix+draft.decode(generated[:i+1],skip_special_tokens=False),add_special_tokens=False).ids
      partial_proposal=partial[committed:committed+min(3,remaining-1)] if partial[:committed]==ids[:committed] else []
      if len(partial_proposal)==min(3,remaining-1):stop='target_limit'
      elif threshold is not None and partial_proposal and confidence[i][3]<threshold:stop='margin'
      if stop is not None:
       require(i+1==len(generated),'continued after first stopping condition');break
     if stop is None:
      if len(generated)==8:stop='draft_budget'
      else:
       require(terminal in inputs['draft_stops'],'unexplained short draft');stop='draft_eos'
     require(trace['stop_reason']==stop,'wrong confidence stopping reason')
     if stop!='draft_eos':require(generated and terminal==generated[-1],'wrong terminal prediction')

     chunks=trace['prefill_chunks'];require(all(c in (1,2,4,8,16) for c in chunks),'unprepared shape')
     remaining_input=trace['missing_tokens'];wanted_chunks=[]
     while remaining_input>0:
      width=16
      if row['mode'] in ('base','scores','stop'):
       width=1
       while width<remaining_input and width<16:width*=2
      wanted_chunks.append(width);remaining_input-=min(width,remaining_input)
     require(chunks==wanted_chunks,'wrong resynchronization profile policy')
     require(sum(chunks)>=trace['missing_tokens'] and (sum(chunks)==0)==(trace['missing_tokens']==0),'missing prefill')
     require(duration(trace['decode_seconds']) and all(duration(s['wall_seconds']) and duration(s['gpu_ms']) for s in trace['decode_samples']),'draft decode time')
     require(trace['prefix_cache_stats']['accounted_host_bytes']<=512*1024*1024 and trace['prefix_cache_stats']['entries']<=2,'unbounded prefix cache')
     executed=trace['executed_draft_ids'];require(executed==generated[:len(executed)] and len(executed)==len(trace['decode_samples']) and len(executed)<=len(generated),'wrong executed draft history')
     if row['mode'] in ('base','scores','stop'):
      require(trace['prefix_source']=='gpu_resident','wrong prefix owner route')
      common=0
      while common<min(len(resident_history),len(prompt)) and resident_history[common]==prompt[common]:common+=1
      kept=common-1 if common==len(prompt) and common<len(resident_history) else common
      stats=trace['reconcile'];resident_calls+=1
      zeroed=(2*draft_layers*draft_heads*(len(resident_history)-kept)*draft_dim+draft_width)*4 if kept<len(resident_history) else 0
      require(stats['calls']==resident_calls and stats['previous_tokens']==len(resident_history),'wrong resident history/counter')
      require(stats['common_tokens']==common and stats['retained_tokens']==kept==trace['cached_tokens'],'wrong resident LCP')
      require(stats['zeroed_bytes']==zeroed and stats['replay_last_token']==(kept<common) and stats['host_snapshot_bytes']==0 and duration(stats['gpu_ms']),'wrong suffix clear/replay')
      require(trace['prefix_cache_stats']['entries']==0 and trace['prefix_cache_stats']['accounted_host_bytes']==0,'resident path captured a host snapshot')
      resident_history=prompt+executed
     else:require(trace['prefix_source']=='host_snapshot' and trace['reconcile'] is None,'wrong snapshot route')

    else:
     require(not proposal,'fallback proposed tokens')
     prefix=target.decode(ids[:committed],skip_special_tokens=False)
     if trace['reason']=='noncanonical_prefix':require(target.encode(prefix,add_special_tokens=False).ids!=ids[:committed],'false noncanonical fallback')
     elif trace['reason']=='draft_capacity':require(len(draft.encode(case['draft_prompt']+prefix,add_special_tokens=False).ids)+8>inputs.get('draft_capacity',inputs['capacity']),'false capacity fallback')
     else:require(remaining==1,'false clipped tail')
    n=0
    for token in proposal:
     if committed+n>=len(ids) or token!=ids[committed+n]:break
     n+=1
    consumed=min(n+1,remaining,len(ids)-committed);require(step['consumed']==consumed and step['accepted_draft']==consumed-1,'wrong acceptance')
    require(step['emitted_ids']==ids[committed:committed+consumed],'wrong emitted target IDs')
    if proposal:
     require(step['submitted_ids']==[ids[committed-1]]+proposal+[0]*(3-len(proposal)),'wrong verifier input')
     metrics=step['metrics']
     for key,wanted in [('target_attention_layout_dispatches',32),('target_attention_vector4_dispatches',32),('target_affine_dispatches',225),('target_affine_misses',0),('target_mpp_affine_dispatches',0),('target_kv_backups',64),('commit_kv_restores',64 if consumed<len(proposal)+1 else 0)]:
      require(metrics[key]==wanted,'wrong native verification route/rollback: '+key)
     if occupancy:
      reuse=len(proposal)==3 or (len(proposal)==2 and occupancy=='occupancy-candidate')
      require(metrics['target_affine_decode_block_dispatches']==(161 if reuse else 1),'wrong three-row projection route')
    else:require(step['submitted_ids']==[ids[committed-1]] and step['metrics'] is None,'fallback target input')
    accepted+=consumed-1;proposed+=len(proposal);committed+=consumed
   require(committed==len(ids),'missing committed output')
  require(all(r['state_sha256']==pair[0]['state_sha256'] and r['hidden_sha256']==pair[0]['hidden_sha256'] for r in pair),'ordinary/speculative final state differs')
  base=next(x for x in pair if x['mode']=='base');scores=base if occupancy else next(x for x in pair if x['mode']=='scores')
  old=witnesses[case['id']]
  require(len(base['rounds'])==len(scores['rounds']),'capture changed target calls')
  if draft_family=='qwen3':require(len(base['rounds'])==len(old['rounds']),'capture changed original target calls')
  # Alternate drafts must preserve target IDs/state, not another model's
  # proposal sequence. The default Qwen3 qualification retains the old witness.
  prior_rounds=old['rounds'] if draft_family=='qwen3' else base['rounds']
  for left,right,prior in zip(base['rounds'],scores['rounds'],prior_rounds):
   for key in ('proposal','submitted_ids','emitted_ids','consumed'):
    require(left[key]==right[key]==prior[key],'capture changed proposals/commits')
   for key in ('generated_ids','executed_draft_ids','prompt_ids'):
    require(left['draft'].get(key)==right['draft'].get(key)==prior['draft'].get(key),'capture changed draft IDs')
  for candidate in pair:
   rounds=candidate['rounds'];accepted=sum(x['accepted_draft'] for x in rounds);proposed=sum(len(x['proposal']) for x in rounds)
   summaries.append(dict(case_id=case['id'],mode=candidate['mode'],base_tps=base['decode_tokens_per_second'],candidate_tps=candidate['decode_tokens_per_second'],ratio=base['decode_seconds']/candidate['decode_seconds'],accepted=accepted,proposed=proposed,target_calls=len(rounds),draft_steps=sum(len(x['draft'].get('executed_draft_ids',[])) for x in rounds),early_margin_stops=sum(x['draft'].get('stop_reason')=='margin' and len(x['draft'].get('generated_ids',[]))<8 for x in rounds),draft_seconds=sum(x['draft']['seconds'] for x in rounds),target_seconds=sum(x['target_seconds'] for x in rounds)))
 return summaries

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args()
 manifest=json.loads(args.manifest.read_text())
 for filename,wanted in manifest['files'].items():require(hashlib.sha256(Path(filename).read_bytes()).hexdigest()==wanted,'changed frozen file: '+filename)
 require(hashlib.sha256(Path(manifest['library']).read_bytes()).hexdigest()==manifest['library_sha256'],'changed native library')
 data=json.loads(args.result.read_text());require(data['library']==manifest['library'],'wrong loaded library')
 fixture=json.loads(args.fixture.read_text());inputs=data['inputs']
 require(fixture['model']==inputs['target_model'] and fixture['stops']==inputs['target_stops'],'changed target identity')
 config=json.loads((Path(inputs['draft_model'])/'tokenizer_config.json').read_text());template=SandboxedEnvironment().from_string(config['chat_template'])
 require([c['id'] for c in fixture['cases'][:8]]==[c['id'] for c in inputs['cases']],'changed fixture coverage')
 for original,case in zip(fixture['cases'][:8],inputs['cases']):
  require(case['rendered']==original['rendered'] and case['prompt_ids']==original['prompt_ids'],'changed fixture prompt')
  require(case['draft_prompt']==template.render(messages=original['messages'],add_generation_prompt=True,enable_thinking=False,tools=None),'changed draft chat prompt')
 require(str(args.reference.resolve()) in manifest['files'],'unbound native reference')
 reference=json.loads(args.reference.read_text())
 require(data['inputs']==json.loads(Path(manifest['argv'][-2]).read_text()),'changed frozen inputs')
 summaries=validate(data,reference);rejected=[]
 mutations={
  'changed_threshold':lambda d:d['inputs'].update(confidence_threshold=2.0),
  'wrong_margin':lambda d:d['records'][1]['rounds'][0]['draft']['confidences'][0].__setitem__(3,999.),
  'wrong_confidence_token':lambda d:d['records'][1]['rounds'][0]['draft']['confidences'][0].__setitem__(0,0),
  'wrong_stop_reason':lambda d:d['records'][1]['rounds'][0]['draft'].update(stop_reason='margin'),
  'incomplete':lambda d:d.update(completed=False),
  'missing_case':lambda d:d['records'].pop(),
  'changed_output':lambda d:d['records'][0]['generated_ids'].__setitem__(12,0),
  'wrong_kv':lambda d:d['records'][1]['state_sha256'][0].__setitem__(0,'0'*64),
  'wrong_hidden':lambda d:d['records'][1].update(hidden_sha256='0'*64),
  'future_cursor':lambda d:d['records'][1]['rounds'][0].update(committed_before=2),
  'wrong_draft_condition':lambda d:d['records'][1]['rounds'][0]['draft']['prompt_ids'].__setitem__(-1,0),
  'wrong_proposal':lambda d:d['records'][1]['rounds'][0]['proposal'].__setitem__(0,0),
  'wrong_acceptance':lambda d:d['records'][1]['rounds'][0].update(consumed=4),
  'missing_rollback':lambda d:d['records'][1]['rounds'][0]['metrics'].update(commit_kv_restores=0),
  'unprepared_shape':lambda d:d['records'][1]['rounds'][0]['draft'].update(prefill_chunks=[7]),
  'wrong_compact_policy':lambda d:d['records'][1]['rounds'][1]['draft'].update(prefill_chunks=[16]),
  'wrong_compact_kv':lambda d:d['records'][1]['state_sha256'][3].__setitem__(1,'0'*64),
  'unbounded_memory':lambda d:d['prepared']['memory'].update(metal_allocated_bytes=10**15),
  'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
  'resident_lcp':lambda d:d['records'][2]['rounds'][0]['draft']['reconcile'].update(common_tokens=1),
  'resident_host_copy':lambda d:d['records'][2]['rounds'][0]['draft']['reconcile'].update(host_snapshot_bytes=1),
  'resident_wrong_counter':lambda d:d['records'][2]['rounds'][0]['draft']['reconcile'].update(calls=99),
  'resident_wrong_executed':lambda d:d['records'][2]['rounds'][0]['draft'].update(executed_draft_ids=[999]),
  'fake_rate':lambda d:d['records'][0].update(decode_tokens_per_second=999),
 }
 for label,mutate in mutations.items():
  changed=copy.deepcopy(data);mutate(changed)
  try:validate(changed,reference)
  except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
  else:raise ValueError('accepted corrupted evidence: '+label)
 with args.output.open('x') as stream:json.dump(dict(completed=True,cases=summaries,rejected_corruptions=rejected,result_sha256=hashlib.sha256(args.result.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__),stream,indent=2)
 print('cross_tokenizer_audit PASS',len(summaries),'case/mode comparisons',len(rejected),'corruptions rejected')
 for r in summaries:print(r['case_id'],r['mode'],round(r['base_tps'],3),round(r['candidate_tps'],3),round(r['ratio'],3),r['accepted'],r['target_calls'],r['draft_steps'])
if __name__=='__main__':main()
