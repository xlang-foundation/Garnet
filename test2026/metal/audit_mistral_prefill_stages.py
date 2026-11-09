"""Unchanged-native diagnostic stage attribution. CPU wall categories are exclusive; GPU/native timing details are nested, never added to wall categories. No speed/coding/frozen-comparator admission."""
from pathlib import Path
import argparse,json,hashlib,math,copy,statistics
import audit_mistral_http_admission
from audit_mistral_http_admission import require,load_client,validate
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study',required=True,type=Path,help='Stage study containing plans1.json and the three serial phase directories')
parser.add_argument('--output',required=True,type=Path,help='Fresh audit JSON path')
parser.add_argument('--fixture',type=Path,help='Defaults to sibling mistral-packed-http1/fixture1.json')
parser.add_argument('--controls',type=Path,help='Defaults to sibling mistral-long-prefix-budget1/http2/campaign/fresh')
args=parser.parse_args()
r=args.study.resolve();out=args.output.resolve();require(not out.exists(),'fresh output required');files={}
def raw(p):
 b=p.read_bytes();files[str(p.resolve())]=hashlib.sha256(b).hexdigest();return b
def read(p):return json.loads(raw(p))
plan,state=read(r/'plans1.json'),read(r/'status1.json');names=['off-direct','on-direct','on-uncached'];require(state['status']=='collected_pending_audit' and [e['id'] for e in state['runs']]==names,'incomplete diagnostic')
fixture=read(args.fixture or r.parent/'mistral-packed-http1/fixture1.json');prompts=fixture['cases'][1]['prompts'];controls={}
for i,label in enumerate(['A','B','C','D']):
 p=(args.controls or r.parent/'mistral-long-prefix-budget1/http2/campaign/fresh')/label/'request-0';wire=[json.loads(x) for x in raw(p.with_suffix('.events.jsonl')).splitlines()];last=next(json.loads(e['data']) for e in reversed(wire) if e['data']!='[DONE]' and 'response_token_ids' in json.loads(e['data']));controls['p2048-c4-o128-'+str(i)]=dict(status='completed',prompt_ids=last['prompt_token_ids'],generated_ids=last['response_token_ids'],text=read(p.with_suffix('.result.json'))['text'])
records=[]
for n in names:
 require((r/(n+'.status')).read_text().strip()=='0','failed phase')
 root=r/n;manifest,status=read(root/'manifest.json'),read(root/'status.json');phase=status['phases']['observed'];require(status['status']=='collected_pending_independent_audit' and phase['status']=='collected','unfinished phase');folder=root/'observed';clients=[]
 for i,e in enumerate(manifest['requests']):
  require(e['label']==['A','B','C','D'][i] and e['body']['max_tokens']==8,'wrong workload');clients.append(load_client(folder/e['label']/'request-0',e['case_id'],e['body'],prompts[i],read,raw))
 require(len(clients)==4,'missing request');events=[json.loads(json.loads(line)['line']) for line in raw(folder/'server/worker-stdout.jsonl').splitlines()]
 records.append(dict(name=n,profile=read(folder/'profile.json'),clients=clients,events=events,shutdown=phase['shutdown'],exit=read(folder/'server/worker-exit.json')))
def interval(e,a='began_monotonic',b='ended_monotonic',duration='run_wall_ms'):
 x,y=e[a],e[b];require(all(type(v) in (float,int) and math.isfinite(v) for v in [x,y,e[duration]]) and 0<x<=y and abs((y-x)*1000-e[duration])<0.01,'bad trace clock/duration');return x,y
def union_ms(spans):
 merged=[]
 for a,b in sorted(spans):
  require(a<=b,'inverted interval')
  if not merged or a>merged[-1][1]:merged.append([a,b])
  else:merged[-1][1]=max(merged[-1][1],b)
 return sum(b-a for a,b in merged)*1000
def trace(events,clients):
 ts=[e for e in events if e['event']=='native_stage_trace'];require(ts,'missing stage trace');allowed={'admit','schedule','prefill','decode','head','prefill_commit','prefill_complete','decode_complete','delivery'};require(all(e['stage'] in allowed for e in ts),'unknown timing category')
 for e in ts:interval(e)
 for stage,count in [('admit',4),('prefill_commit',62),('prefill_complete',4),('decode_complete',28),('delivery',32),('head',2)]:require(sum(e['stage']==stage for e in ts)==count,'wrong '+stage+' timing coverage')
 batches=[e for e in events if e['event']=='native_batch'];runs=[e for e in ts if e['stage'] in ('prefill','decode')];require(len(runs)==len(batches),'missing native run timing')
 for e,b in zip(runs,batches):
  require(e['stage']==b['stage'] and e['active_tokens']==sum(z['prefill_tokens'] if b['stage']=='prefill' else 1 for z in b['requests']),'wrong timing/run binding')
  require(e['partial_submissions']==0,'overlapped partial submissions unqualified')
  for k in ['native_inference_ms','prepare_ms','encode_ms','wait_ms','gpu_ms','embedding_wall_ms']:require(math.isfinite(e[k]) and e[k]>=0,'invalid native time')
  require(e['gpu_ms']>0 and e['gpu_ms']<=e['wait_ms']+1 and e['native_inference_ms']<=e['run_wall_ms']+1,'stale/impossible native GPU/wall time')
  require(e['prepare_ms']+e['encode_ms']+e['wait_ms']<=e['native_inference_ms']+1,'double-counted native phases')
  a,b=interval(e,'embedding_began_monotonic','embedding_ended_monotonic','embedding_wall_ms');require(b<=e['began_monotonic'],'embedding overlaps execution')
 for e in ts:
  if e['stage']=='head':require(0<e['gpu_ms']<=e['run_wall_ms']+1,'stale head GPU time')
 first=min(e['began_monotonic'] for e in ts if e['stage']=='schedule');last=max(e['ended_monotonic'] for e in ts if e['stage']=='prefill_complete');total=(last-first)*1000;require(total>0,'empty prefill window')
 categories={k:[] for k in ['prefill_calls','decode_calls','head_calls','embedding','scheduling','native_progress_and_cache_commits','delivery']};gpu=0;details=[]
 for e in ts:
  a,b=interval(e);stage=e['stage'];category={'prefill':'prefill_calls','decode':'decode_calls','head':'head_calls','schedule':'scheduling','prefill_commit':'native_progress_and_cache_commits','prefill_complete':'native_progress_and_cache_commits','decode_complete':'native_progress_and_cache_commits','delivery':'delivery'}.get(stage)
  if category and a<last and b>first:
   require(first<=a<=b<=last,'partial diagnostic interval crosses window');categories[category].append((a,b))
   if stage in ('prefill','decode','head'):gpu+=e['gpu_ms']
  if stage in ('prefill','decode'):
   a,b=interval(e,'embedding_began_monotonic','embedding_ended_monotonic','embedding_wall_ms')
   if a<last and b>first:require(first<=a<=b<=last,'partial embedding interval');categories['embedding'].append((a,b))
  if stage=='prefill':details.append({k:e[k] for k in ['active_tokens','prepare_ms','encode_ms','wait_ms','gpu_ms','run_wall_ms']})
 spans=[z for vs in categories.values() for z in vs];summed=sum(union_ms(z) for z in categories.values());covered=union_ms(spans);require(abs(summed-covered)<0.01 and covered<=total+0.01,'overlapping wall categories');require(0<gpu<=union_ms(categories['prefill_calls']+categories['decode_calls']+categories['head_calls'])+1,'GPU duration exceeds containing calls')
 main=[e for e in details if e['active_tokens']==512];require(len(main)==16 and len(details)==17,'wrong full/tail prefill coverage')
 requests=[]
 for c in clients:
  admission=next(e for e in ts if e['stage']=='admit' and e['id']==c['id']);sent=[e for e in ts if e['stage']=='delivery' and e['id']==c['id']];require(sorted(e['sampled_tokens'] for e in sent)==list(range(1,9)),'delivery sequence differs');first_sent=next(e for e in sent if e['sampled_tokens']==1);requests.append(dict(id=c['id'],native_request=admission['native_request'],worker_admit_to_first_token_seconds=first_sent['ended_monotonic']-admission['began_monotonic'],client_ttft_seconds=c['result']['ttft_content_seconds']))
 return dict(prefill_window_wall_ms=total,exclusive_wall_ms={k:union_ms(v) for k,v in categories.items()},unassigned_wall_ms=total-covered,gpu_command_span_ms=gpu,gpu_fraction_of_window=gpu/total,main_chunk_medians_ms={k:statistics.median(e[k] for e in main) for k in ['prepare_ms','encode_ms','wait_ms','gpu_ms','run_wall_ms']},prefill_run_details=details,requests=requests)
def check(records):
 results={};canonical=None
 for i,run in enumerate(records):
  trace_on=i!=0;cached=i!=2;require(run['profile']['trace_stage_latency'] is trace_on and run['profile']['prefix_cache'] is cached,'wrong profile')
  require(run['shutdown']['graceful_native_closed'] and not run['shutdown']['forced_kill'],'unclean shutdown')
  native=validate(run['events'],run['clients'],controls,run['profile'],run['exit'],expected_requests=4,control_prefixes=True,maximum_chunk=128,prefix_policy=cached,required_active_rows=4)
  observed=[dict(ids=c['sse_ids'],text=c['sse_text'],finish=c['sse_finish']) for c in run['clients']];require(all(len(c['sse_ids'])==8 and c['result']['cached_prompt_tokens']==0 for c in run['clients']),'wrong cold output/cache coverage')
  if canonical is None:canonical=observed
  else:require(observed==canonical,'trace/cache numerical output mismatch')
  if trace_on:results[run['name']]=trace(run['events'],run['clients'])
  else:require(not any(e['event']=='native_stage_trace' for e in run['events']),'disabled tracing emitted timings')
 return results
results=check(records);rejected=[]
for label in ['missing-commit','wrong-duration','fake-gpu','stale-head','overlap','wrong-token','wrong-epoch','fake-partial-submit']:
 bad=copy.deepcopy(records);ts=[e for e in bad[1]['events'] if e['event']=='native_stage_trace']
 if label=='missing-commit':bad[1]['events'].remove(next(e for e in ts if e['stage']=='prefill_commit'))
 elif label=='wrong-duration':next(e for e in ts if e['stage']=='prefill')['run_wall_ms']+=100
 elif label=='fake-gpu':next(e for e in ts if e['stage']=='prefill')['gpu_ms']=1e9
 elif label=='stale-head':next(e for e in ts if e['stage']=='head')['gpu_ms']=1e9
 elif label=='overlap':e=next(e for e in ts if e['stage']=='prefill_commit');other=next(e for e in ts if e['stage']=='prefill');e.update(began_monotonic=other['began_monotonic'],ended_monotonic=other['ended_monotonic'],run_wall_ms=other['run_wall_ms'])
 elif label=='wrong-token':bad[2]['clients'][0]['sse_ids'][0]=-1
 elif label=='wrong-epoch':next(e for e in bad[1]['events'] if e['event']=='native_batch')['requests'][0]['epoch']=0
 else:next(e for e in ts if e['stage']=='prefill')['partial_submissions']=1
 try:check(bad)
 except (ValueError,KeyError,IndexError,StopIteration):rejected.append(label)
 else:raise ValueError('accepted '+label)
require(files.get(str(Path(plan['native_library']).resolve()),plan['native_library_sha256'])==plan['native_library_sha256'],'library metadata mismatch');raw(Path(plan['native_library']));require(files[str(Path(plan['native_library']).resolve())]==plan['native_library_sha256'],'changed native library')
for p in [Path(__file__),Path(audit_mistral_http_admission.__file__)]:raw(p)
for p in [r/'source1/test2026/metal/local_coder_batch_worker.py',r/'tools/local_coder_batch_worker.py',r/'tools/run_mistral_cold_prefix.py']:raw(p)
out.write_text(json.dumps(dict(passed=True,scope=__doc__,results=results,paired_output_token_observations=96,rejected_corruptions=rejected,files_sha256=files,exclusive_categories_not_added_to_nested_gpu_times=True,no_serving_speed_or_coding_admission=True),indent=2)+'\n');print(json.dumps({k:{p:v[p] for p in ['prefill_window_wall_ms','exclusive_wall_ms','unassigned_wall_ms','gpu_command_span_ms','gpu_fraction_of_window']} for k,v in results.items()},indent=2));print('stage audit PASS',len(rejected),'corruptions')
