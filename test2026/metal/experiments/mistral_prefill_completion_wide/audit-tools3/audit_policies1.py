import json,math,statistics
from audit_half_transport1 import require

def split_policy(events,profile):
 prepared=[e for e in events if e['event']=='native_split_score_prepared'];dispatches=[e for e in events if e['event']=='native_split_score_dispatch'];enabled=profile['split_decode_scores'];need_budget=profile['affine_weight_residency_budget_bytes'];require(type(enabled) is bool and need_budget==0,'wrong score/residency policy')
 require(len(prepared)==1 and prepared[0]==dict(event='native_split_score_prepared',enabled=enabled,chunk=128,minimum_visible=512),'wrong score preparation')
 ready=next(e for e in events if e['event']=='ready');backend={p.strip() for p in ready['backend'].split(';')};require({'split_decode_scores_optional','split_decode_scores_chunk=128','split_decode_scores_min_visible=512'}<=backend,'missing compiled score policy')
 require(events.index(prepared[0])<events.index(ready),'score preparation after readiness')
 affine=[e for e in events if e['event']=='native_affine_profile'];require(dispatches and len(dispatches)==len(affine),'missing score dispatch coverage')
 for at,d in enumerate(events):
  if d['event']!='native_split_score_dispatch':continue
  require(d['enabled'] is enabled and d['stage'] in ['prefill','decode','head'] and d['dispatches']==(32 if enabled and d['stage']=='decode' else 0),'wrong actual score route/count')
  require(at+1<len(events) and events[at+1]['event']=='native_affine_profile' and events[at+1]['stage']==d['stage'],'score observation not bound to native kernel frame')
 require(sum(e['stage']=='decode' for e in dispatches)==sum(e['event']=='native_batch' and e['stage']=='decode' for e in events),'decode score observations do not cover all native batches')
 return dict(enabled=enabled,prepared=prepared[0],observed_decode_frames=sum(e['stage']=='decode' for e in dispatches),selected_layer_dispatches=sum(e['dispatches'] for e in dispatches))

def profile32_policy(events,profile):
 enabled=profile['split_score_profile32'];require(type(enabled) is bool and profile['split_decode_scores'] is True,'wrong profile32 policy')
 prepared=[e for e in events if e['event']=='native_profile32_score_prepared'];require(prepared==[dict(event='native_profile32_score_prepared',enabled=enabled,capacity=4096,heads=32,kv_heads=8,dim=128)],'wrong profile32 preparation')
 ready=next(e for e in events if e['event']=='ready');require('split_score_profile32_optional' in {v.strip() for v in ready['backend'].split(';')},'missing compiled profile32 support')
 require(events.index(prepared[0])<events.index(ready),'late profile32 preparation')
 dispatches=[e for e in events if e['event']=='native_profile32_score_dispatch'];require(len(dispatches)==sum(e['event']=='native_split_score_dispatch' for e in events),'missing profile32 observation')
 for at,d in enumerate(events):
  if d['event']!='native_profile32_score_dispatch':continue
  require(d['enabled'] is enabled and d['stage'] in ['prefill','decode','head'] and d['dispatches']==(32 if enabled and d['stage']=='decode' else 0),'wrong profile32 native route')
  require(at+2<len(events) and events[at+1]['event']=='native_split_score_dispatch' and events[at+2]['event']=='native_affine_profile' and events[at+1]['stage']==events[at+2]['stage']==d['stage'],'profile32 observation not bound to actual native frame')
 require(sum(d['stage']=='decode' for d in dispatches)==sum(e['event']=='native_batch' and e['stage']=='decode' for e in events),'profile32 decode coverage differs')
 return dict(enabled=enabled,observed_decode_frames=sum(d['stage']=='decode' for d in dispatches),selected_layer_dispatches=sum(d['dispatches'] for d in dispatches))

def value_policy(events,profile):
 enabled=profile['value_profile32'];require(type(enabled) is bool and profile['split_score_profile32'] is True and profile['split_decode_scores'] is True,'wrong value/score policy')
 prepared=[e for e in events if e['event']=='native_value_profile32_prepared'];require(prepared==[dict(event='native_value_profile32_prepared',enabled=enabled,capacity=4096,heads=32,kv_heads=8,dim=128,threads=128)],'wrong value-profile preparation')
 ready=next(e for e in events if e['event']=='ready');require({'value_profile32_optional','value_profile32_threads=128'}<={v.strip() for v in ready['backend'].split(';')},'missing native value-profile support')
 require(events.index(prepared[0])<events.index(ready),'late value-profile preparation')
 dispatches=[e for e in events if e['event']=='native_value_profile32_dispatch'];require(len(dispatches)==sum(e['event']=='native_affine_profile' for e in events),'missing value-profile frame')
 for at,d in enumerate(events):
  if d['event']!='native_value_profile32_dispatch':continue
  require(d['enabled'] is enabled and d['stage'] in ['prefill','decode','head'] and d['dispatches']==(32 if enabled and d['stage']=='decode' else 0),'wrong native value route/count')
  require(at+3<len(events) and [e['event'] for e in events[at+1:at+4]]==['native_profile32_score_dispatch','native_split_score_dispatch','native_affine_profile'] and all(e['stage']==d['stage'] for e in events[at+1:at+4]),'value observation not bound to native frame')
 require(sum(e['stage']=='decode' for e in dispatches)==sum(e['event']=='native_batch' and e['stage']=='decode' for e in events),'missing native value decode coverage')
 return dict(enabled=enabled,observed_decode_frames=sum(e['stage']=='decode' for e in dispatches),selected_layer_dispatches=sum(e['dispatches'] for e in dispatches))

def paired_policy(events,profile):
 enabled=profile['paired_prefill'];require(type(enabled) is bool and profile['value_profile32'] is True,'wrong pair/value policy')
 prepared=[e for e in events if e['event']=='native_paired_prefill_prepared'];require(prepared==[dict(event='native_paired_prefill_prepared',enabled=enabled,maximum_rows=512,threads=256)],'wrong pair preparation')
 ready=next(e for e in events if e['event']=='ready');require('paired_packed_prefill_optional' in {v.strip() for v in ready['backend'].split(';')},'missing pair support')
 require(events.index(prepared[0])<events.index(ready),'late pair preparation')
 dispatches=[e for e in events if e['event']=='native_paired_prefill_dispatch'];require(len(dispatches)==sum(e['event']=='native_affine_profile' for e in events),'missing paired frame')
 for at,d in enumerate(events):
  if d['event']!='native_paired_prefill_dispatch':continue
  require(at+4<len(events) and [e['event'] for e in events[at+1:at+5]]==['native_value_profile32_dispatch','native_profile32_score_dispatch','native_split_score_dispatch','native_affine_profile'] and all(e['stage']==d['stage'] for e in events[at+1:at+5]),'pair observation not bound to native frame')
  affine=events[at+4];expected=32 if enabled and d['stage']=='prefill' and affine['active_rows']>=128 else 0
  require(d['enabled'] is enabled and d['stage'] in ['prefill','decode','head'] and d['dispatches']==expected,'wrong native pair route/count')
 return dict(enabled=enabled,observed_frames=len(dispatches),selected_layer_pairs=sum(e['dispatches'] for e in dispatches))

def silu_policy(events,profile):
 enabled=profile['silu_profile32'];require(type(enabled) is bool and profile['paired_prefill'] is False,'wrong SiLU/pair policy')
 prepared=[e for e in events if e['event']=='native_silu_profile32_prepared'];require(prepared==[dict(event='native_silu_profile32_prepared',enabled=enabled,maximum_rows=512,columns=[4096,14336],threads=128)],'wrong SiLU preparation')
 ready=next(e for e in events if e['event']=='ready');require({'silu_profile32_optional','silu_profile32_max_rows=512','silu_profile32_threads=128'}<={v.strip() for v in ready['backend'].split(';')},'missing SiLU support')
 require(events.index(prepared[0])<events.index(ready),'late SiLU preparation')
 dispatches=[e for e in events if e['event']=='native_silu_profile32_dispatch'];require(len(dispatches)==sum(e['event']=='native_affine_profile' for e in events),'missing SiLU frame')
 for at,d in enumerate(events):
  if d['event']!='native_silu_profile32_dispatch':continue
  require(at+5<len(events) and [e['event'] for e in events[at+1:at+6]]==['native_paired_prefill_dispatch','native_value_profile32_dispatch','native_profile32_score_dispatch','native_split_score_dispatch','native_affine_profile'] and all(e['stage']==d['stage'] for e in events[at+1:at+6]),'SiLU observation not bound to native frame')
  expected=32 if enabled and d['stage'] in ['prefill','decode'] else 0
  require(d['enabled'] is enabled and d['stage'] in ['prefill','decode','head'] and d['dispatches']==expected,'wrong native SiLU route/count')
 return dict(enabled=enabled,observed_frames=len(dispatches),selected_layer_dispatches=sum(e['dispatches'] for e in dispatches))

def timing(clients):
 for c in clients:
  d=c['result'];need_keys=['submitted','completed','first_content','last_content','first_usage_arrival','last_usage_arrival','ttft_content_seconds','decode_tokens_per_second']
  require(all(isinstance(d[k],(int,float)) and math.isfinite(d[k]) for k in need_keys),'nonfinite timing')
  require(d['submitted']<=d['first_content']<=d['last_content']<=d['completed'] and d['generated_tokens']==128,'invalid time/count order')
  require(math.isclose(d['ttft_content_seconds'],d['first_content']-d['submitted'],rel_tol=2e-12,abs_tol=1e-9),'wrong TTFT arithmetic')
  require(d['last_usage_arrival']>d['first_usage_arrival'] and d['first_usage_batch_tokens']==1,'invalid usage timing')
  require(math.isclose(d['decode_tokens_per_second'],127/(d['last_usage_arrival']-d['first_usage_arrival']),rel_tol=2e-12),'wrong decode arithmetic')
  wire=c['wire'];content=[e['monotonic'] for e in wire if e['data']!='[DONE]' and any(v.get('delta',{}).get('content') for v in json.loads(e['data']).get('choices',[]))]
  require(content and content[0]==d['first_content'] and content[-1]==d['last_content'],'TTFT/content not bound to raw SSE')
  usage=[e['monotonic'] for e in wire if e['data']!='[DONE]' and json.loads(e['data']).get('usage',{}).get('completion_tokens',0)>0 and 'token_id' in json.loads(e['data'])]
  require(usage and usage[0]==d['first_usage_arrival'] and usage[-1]==d['last_usage_arrival'],'decode not bound to raw SSE')
 return dict(ttft_median_seconds=statistics.median(c['result']['ttft_content_seconds'] for c in clients),decode_median_tps=statistics.median(c['result']['decode_tokens_per_second'] for c in clients),aggregate_tps=(128*len(clients))/(max(c['result']['completed'] for c in clients)-min(c['result']['submitted'] for c in clients)))
