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
def validate_trace(inputs,case,ids,rounds,resident_calls):
 target=Tokenizer.from_file(inputs['target_model']+'/tokenizer.json');draft=Tokenizer.from_file(inputs['draft_model']+'/tokenizer.json')
 draft_layers,draft_heads,draft_dim,draft_width=24,2,64,896
 occupancy='occupancy-control';row=dict(mode='base',rounds=rounds)
 committed=1;hits=0;accepted=0;proposed=0;resident_history=[]
 for step in row['rounds']:
  require(step['committed_before']==committed and step['slot']==len(case['prompt_ids'])+committed-1,'noncausal cursor')
  proposal=step['proposal'];trace=step['draft'];remaining=inputs['output_limit']-committed
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
  else:
   require(step['submitted_ids']==[ids[committed-1],0,0,0],'scheduled single-token verification input')
   metrics=step['metrics']
   for key,wanted in [('target_attention_layout_dispatches',32),('target_attention_vector4_dispatches',32),('target_affine_dispatches',225),('target_affine_misses',0),('target_mpp_affine_dispatches',0),('target_kv_backups',64),('commit_kv_restores',0),('target_affine_decode_block_dispatches',1)]:
    require(metrics[key]==wanted,'scheduled single-token verification route: '+key)
  accepted+=consumed-1;proposed+=len(proposal);committed+=consumed
 require(committed==len(ids),'missing committed output')
 return resident_calls,dict(accepted=accepted,proposed=proposed,target_calls=len(rounds))
