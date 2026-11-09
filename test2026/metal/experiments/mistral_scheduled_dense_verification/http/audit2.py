"""Audit diagnostic B1 HTTP streams, native scheduler progress and helper causality."""
from pathlib import Path
import copy,hashlib,json,math,statistics,sys
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
R=Path(__file__).resolve().parent;M=R.parent;SG=M/'mistral-fused-mlp-serving1';B=M/'mistral-bounded-draft-long1';C=M/'mistral-scheduled-dense-verification1'
sys.path.insert(0,str(SG/'audit-tools1'));from audit_half_transport1 import load_client,require
from audit_policies1 import timing
sys.path.insert(0,str(R));from audit_trace1 import validate_trace
plain=lambda p:json.loads(p.read_text());files={}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def raw(p):
 a=p.read_bytes();files[str(p)]=hashlib.sha256(a).hexdigest();return a
def read(p):return json.loads(raw(p))
PROFILE=plain(R/'profile1.json');INPUTS=PROFILE['runtime_inputs'];CONTROL=plain(B/'frozen-control1.json');GOLD=plain(B/'qualification1-result.json')['records'][0]['generated_ids']
plan=next(p for p in plain(SG/'serving-manifest1.json')['plans'] if p['id']=='p2048-c1-o128');BODY=plain(Path(plan['cases']))[0];PROMPT=plain(Path(plan['workload']))['requests'][0]['token_ids']
def check_request(events,client,starting_calls):
 key=client['id'];selected=[e for e in events if e.get('id')==key];admit=[e for e in selected if e['event']=='native_admit'];done=[e for e in selected if e['event']=='completed'];tokens=[e for e in selected if e['event']=='token'];steps=[e for e in selected if e['event']=='native_dense_verification']
 require(len(admit)==len(done)==1 and len(tokens)==128 and len(steps)>0,'missing native events')
 a,d=admit[0],done[0];require(a['prompt_ids']==d['prompt_ids']==PROMPT and a['limit']==128 and d['generated_ids']==GOLD,'native target IDs differ')
 require(client['sse_ids']==client['final_response_ids']==GOLD and client['final_prompt_ids']==PROMPT and client['sse_done']==1 and client['stream_cached_counts']==[0] and client['sse_finish']=='length','raw client IDs/cache/finish differ')
 require(client['body']==BODY and client['result']['status']=='completed' and client['result']['cached_prompt_tokens']==0 and client['result']['prompt_tokens']==2048 and client['result']['generated_tokens']==128,'client workload changed')
 require(d['cached_tokens']==0 and d['finish_reason']=='length' and d['generation_mode']=='native_scheduled_dense_speculation_B1','wrong serving mode')
 require([x['token_id'] for x in tokens]==GOLD and [x['sampled_tokens'] for x in tokens]==list(range(1,129)) and all(x['prompt_tokens']==2048 and x['cached_tokens']==0 for x in tokens),'native token stream differs')
 require(d['text']==client['sse_text']==client['result']['text'],'HTTP text differs')
 tok=Tokenizer.from_file(INPUTS['draft_model']+'/tokenizer.json');tc=plain(Path(INPUTS['draft_model'])/'tokenizer_config.json');template=SandboxedEnvironment().from_string(tc['chat_template']);user=BODY['messages'][0]['content'];tail=tok.decode(tok.encode(user,add_special_tokens=False).ids[-192:],skip_special_tokens=False);expected=template.render(messages=[dict(role='user',content=tail)],add_generation_prompt=True,enable_thinking=False,tools=None)
 require(a['draft_prompt']==expected,'helper prompt was changed or not bounded')
 rounds=[];request_id=d['native_request'];epoch=None
 for e in steps:
  native=e['native'];require(len(native['requests'])==1,'native dense row count');q=native['requests'][0];p=e['native_schedule'];require(p['stage']=='decode' and len(p['requests'])==1,'native plan differs');planned=p['requests'][0]
  if epoch is None:epoch=q['epoch']
  count=e['committed_before'];require(q['id']==planned['id']==request_id and q['row']==planned['row']==0 and q['epoch']==planned['epoch']==epoch and p['epochs']==[epoch] and p['active_rows']==[1],'native lease differs')
  require(q['known']==planned['token_id']==GOLD[count-1] and planned['generated_tokens']==count and q['slot_before']==planned['slot']==e['slot']==2048+count-1 and q['remaining_before']==128-count and q['submitted']==len(e['proposal'])+1,'native known/slot/budget differs')
  require(q['consumed']==len(q['emitted']) and q['finished']==(count+q['consumed']==128) and not q['eos'],'native terminal budget differs')
  metrics=native['verification'];require(metrics['target_attention_split_score_dispatches']==32 and metrics['consumed_inputs']==[q['consumed']] and metrics['emitted_tokens']==[q['emitted']],'new split verifier did not execute/commit')
  rounds.append(dict(committed_before=count,slot=e['slot'],proposal=e['proposal'],submitted_ids=e['submitted_ids'],emitted_ids=q['emitted'],consumed=q['consumed'],accepted_draft=q['consumed']-1,target_seconds=e['target_seconds'],draft=e['draft'],metrics=metrics))
 calls,summary=validate_trace(INPUTS,dict(draft_prompt=a['draft_prompt'],prompt_ids=PROMPT),GOLD,rounds,starting_calls)
 require(d['speculative_metrics']==dict(target_calls=summary['target_calls'],accepted=summary['accepted'],proposed=summary['proposed']),'native speculative totals differ')
 require(all(d['native_status'][k]==0 for k in ['active','queued','inflight','tracked']),'terminal native request remains')
 ticks=d['prefill_ticks'];require(len(ticks)==16,'missing native prefill ticks')
 for i,t in enumerate(ticks):
  p=t['plan'];r=p['requests'][0];require(p['stage']=='prefill' and len(p['requests'])==1 and r['id']==request_id and r['epoch']==epoch and r['row']==0 and r['cached_tokens']==0 and r['prefill_offset']==i*128 and r['prefill_tokens']==128 and t['token_count']==(i+1)*128 and t['final']==(i==15),'native prefill progress differs')
 return calls,summary

def main():
 require(not (R/'audit2.json').exists(),'audit overwrite');pre=read(R/'preflight1.json');require(pre['passed'] and read(R/'postflight1.json')['passed'] and read(R/'result1.json')['completed'] and read(C/'qualification-audit2.json')['passed'],'collection unfinished')
 for n,h in pre['files_sha256'].items():require(sha(Path(n))==h,'admitted file changed '+n)
 metrics=[];last=0;pids=set();fixtures=[];total_steps=0
 for engine in [1,2]:
  root=R/f'engine{engine}';status=read(root/'status1.json');require(status['status']=='terminal' and status['returncode']==0 and status['pid'] not in pids and status['finished_ns']>status['started_ns']>=last,'engines not fresh/serial/terminal');last=status['finished_ns'];pids.add(status['pid'])
  events=[json.loads(json.loads(line)['line']) for line in raw(root/'server/worker-stdout.jsonl').splitlines()];ready=[e for e in events if e['event']=='ready'];closed=[e for e in events if e['event']=='closed']
  require(len(ready)==len(closed)==1 and ready[0]['active_requests']==1 and ready[0]['profile']==PROFILE and ready[0]['target_graphs']==224 and ready[0]['draft_graphs']==96,'engine profile differs')
  require('native_scheduled_dense_verify_v1' in ready[0]['backend'] and 'native_mistral_verification_split_scores128_default_v1' in ready[0]['backend'],'wrong native build')
  require(read(root/'server/worker-exit.json')['returncode']==0 and closed[0]['memory']['metal_allocated_bytes']<=4194304 and all(closed[0]['memory'][k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'native ownership cleanup differs')
  require(len([e for e in events if e['event']=='completed'])==3 and not any(e['event']=='error' for e in events),'native request errors/count')
  calls=0
  for wave in [0,1,2]:
   prefix=root/f'wave{wave}/request-0';c=load_client(prefix,'p2048-c1-o128',BODY,PROMPT,read,raw);c['wire']=[json.loads(line) for line in raw(prefix.with_suffix('.events.jsonl')).splitlines()]
   require(all(b['monotonic']>=a['monotonic'] for a,b in zip(c['wire'],c['wire'][1:])),'SSE clocks not monotonic')
   calls,summary=check_request(events,c,calls);total_steps+=summary['target_calls'];x=timing([c]);metrics.append(dict(engine=engine,wave=wave,phase='cold' if wave==0 else 'warmed',**x));fixtures.append((events,c,calls))
 warm=[r for r in metrics if r['phase']=='warmed'];med={k:statistics.median(r[k] for r in warm) for k in ['ttft_median_seconds','decode_median_tps','aggregate_tps']};reference=read(SG/'comparison1.json')['references'];ratios={name:dict(ttft_native_over_reference=med['ttft_median_seconds']/values['p2048-c1-o128']['median']['ttft_median_seconds'],decode_native_over_reference=med['decode_median_tps']/values['p2048-c1-o128']['median']['decode_median_tps'],aggregate_native_over_reference=med['aggregate_tps']/values['p2048-c1-o128']['median']['aggregate_tps']) for name,values in reference.items()}
 # Corrupt native and client evidence independently; these checks do not rerun inference.
 events,c,_=fixtures[0];rejected=[]
 mutations={
  'fake_budget':lambda e,c:next(x for x in e if x['event']=='native_dense_verification' and x['id']==c['id'])['native']['requests'][0].update(remaining_before=999),
  'wrong_epoch':lambda e,c:next(x for x in e if x['event']=='native_dense_verification' and x['id']==c['id'])['native']['requests'][0].update(epoch=999),
  'future_helper_prefix':lambda e,c:next(x for x in e if x['event']=='native_dense_verification' and x['id']==c['id'] and x['draft'].get('prompt_ids'))['draft']['prompt_ids'].append(1),
  'missing_rollback':lambda e,c:next(x['native']['verification'] for x in e if x['event']=='native_dense_verification' and x['id']==c['id'] and x['native']['requests'][0]['consumed']<len(x['proposal'])+1).update(commit_kv_restores=0),
  'fake_split_counter':lambda e,c:next(x for x in e if x['event']=='native_dense_verification' and x['id']==c['id'])['native']['verification'].update(target_attention_split_score_dispatches=0),
  'wrong_stream_id':lambda e,c:c['sse_ids'].__setitem__(0,-1),
  'false_ttft':lambda e,c:c['result'].update(ttft_content_seconds=.001),
  'false_decode':lambda e,c:c['result'].update(decode_tokens_per_second=999.),
 }
 for name,fn in mutations.items():
  e2,c2=copy.deepcopy(events),copy.deepcopy(c);fn(e2,c2)
  try:check_request(e2,c2,0);timing([c2])
  except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
  else:raise ValueError('corruption admitted '+name)
 report=dict(passed=True,scope=__doc__,requests=6,output_ids=768,all_output_ids_exact_current_native=True,native_target_calls=total_steps,split_layer_dispatches=32*total_steps,metrics=metrics,warmed_median=med,diagnostic_ratios_vs_frozen_references=ratios,negative_controls_rejected=rejected,files_sha256=files,http_full_kv_hidden_observed=False,separate_native_full_state_qualification=str(C/'qualification-audit2.json'),active_rows=1,reference_max_sequences=4,matched_maximum_concurrency_profile=False,c4_throughput_measured=False,selected_full_serving_comparison_promoted=False,coding_quality_qualified=False,no_comparator_execution=True,goal_complete=False)
 (R/'audit2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
