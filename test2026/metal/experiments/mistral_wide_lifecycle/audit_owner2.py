"""Audit exact C4/T4 current-ordinary parity and independent native row lifecycle."""
from pathlib import Path
import copy,hashlib,json
R=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
def need(v,why):
 if not v:raise ValueError(why)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
LABELS=['initial_prefilled','finish_A','claim_E','mid_E','cancel_C','claim_F','finish_E','finish_B','before_late_G','all_done']
def check(a,b):
 for d,mode in [(a,'ordinary'),(b,'verify')]:
  need(d['completed'] and d['mode']==mode and d['inputs']==read(R/'input2.json'),'wrong arm/input')
  need('native_scheduled_dense_verify_v1' in d['backend'] and len(d['jobs'])==7,'wrong native engine')
  memory=d['prepared']['memory'];need(memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-536870912 and d['prepared']['target_graphs']==224,'prepared memory/profile')
  end=d['cleanup'];need(end['metal_allocated_bytes']<=4194304 and all(end[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'cleanup ownership')
  jobs={j['label']:j for j in d['jobs']};need(set(jobs)==set('ABCDEFG'),'request coverage')
  for item in d['inputs']['jobs']+[d['inputs']['late_probe']]:
   j=jobs[item['label']];need(all(j[k]==item[k] for k in ['label','prompt_ids','limit']) and j['prefilled']==len(j['prompt_ids']),'prompt/budget changed')
   need(j['status']==('cancelled' if j['label']=='C' else 'finished') and len(j['generated'])==(12 if j['label']=='C' else j['limit']),'native terminal/cancel count')
  need(jobs['A']['row']==jobs['E']['row'] and jobs['E']['epoch']>jobs['A']['epoch'] and jobs['C']['row']==jobs['F']['row'] and jobs['F']['epoch']>jobs['C']['epoch'],'native row reuse/epoch')
  need(len({jobs[x]['row'] for x in 'ABCD'})==4,'initial rows not independent')
 need(a['jobs']==b['jobs'],'native request outputs/placements differ from ordinary')
 need(len(a['ticks'])==len(b['ticks']),'tick coverage changed')
 native_calls=0;active_counts=set();mixed=0
 for x,y in zip(a['ticks'],b['ticks']):
  need(x['plan']==y['plan'],'native schedule differs from ordinary');need(len(x['executions'])==len(y['executions'])==len(x['plan']['batches']),'execution coverage')
  if len(x['plan']['batches'])>1:mixed+=1
  for p,r,s in zip(x['plan']['batches'],x['executions'],y['executions']):
   need(r['stage']==s['stage']==p['stage'],'wrong execution stage')
   if p['stage']=='prefill':need(r==s,'native prefill changed');continue
   native=s['native'];need(r['native'] is None and native is not None,'reference used verifier or candidate skipped it')
   q=native['requests'];need(len(q)==len(p['requests'])==len(r['completed'])==len(s['completed']),'native active row count');active_counts.add(len(q));native_calls+=1
   metrics=native['verification'];need(metrics['target_affine_dispatches']==225 and metrics['target_affine_misses']==0 and metrics['target_attention_dispatches']==32 and metrics['target_attention_layout_dispatches']==0 and metrics['target_attention_split_score_dispatches']==0 and metrics['target_kv_backups']==64 and metrics['commit_kv_restores']==(64 if any(q['submitted']>1 for q in q) else 0) and metrics['target_indexed_decode_dispatches']==225 and metrics['target_mpp_affine_dispatches']==0 and metrics['target_indexed_prefill_dispatches']==0,'native B4/T4 route differs')
   input_jobs={j['native_request']:j for j in b['jobs']}
   for expected,reference,actual in zip(p['requests'],r['completed'],q):
    j=input_jobs[actual['id']];need(actual['id']==reference['id']==expected['id'] and actual['row']==reference['row']==expected['row'] and actual['epoch']==reference['epoch']==expected['epoch'],'native identity/lease')
    need(actual['known']==expected['token_id'] and actual['slot_before']==expected['slot'] and actual['remaining_before']==j['limit']-expected['generated_tokens'] and actual['consumed']==1 and actual['submitted']==min(4,actual['remaining_before']),'native known/slot/budget')
    need(actual['emitted']==reference['emitted']==j['generated'][expected['generated_tokens']:expected['generated_tokens']+1] and actual['finished']==reference['finished'] and not actual['eos'],'native emitted/terminal span differs')
 need(active_counts=={1,2,3,4} and mixed>=16,'missing occupancy or mixed-prefill lifecycle')
 aw={w['label']:w for w in a['witnesses']};bw={w['label']:w for w in b['witnesses']}
 need(list(aw)==LABELS and set(bw)==set(LABELS)|{'before_invalid_mixed','after_invalid_mixed','before_invalid_budget','after_invalid_budget'},'missing lifecycle witness')
 for label in LABELS:
  x,y=aw[label],bw[label];need(x==y,'complete four-row state/status/owners differ '+label)
  need(len(x['states'])==32 and all(len(layer)==4 and all(len(row)==2 and all(isinstance(h,str) and len(h)==64 for h in row) for row in layer) for layer in x['states']) and len(x['hidden'])==4,'incomplete full-state witness')
 before=copy.deepcopy(bw['before_invalid_mixed']);after=copy.deepcopy(bw['after_invalid_mixed']);before.pop('label');after.pop('label');need(before==after,'rejected mixed call changed full state')
 one=copy.deepcopy(bw['before_invalid_budget']);two=copy.deepcopy(bw['after_invalid_budget']);one.pop('label');two.pop('label');need(one==two,'invalid native budget changed full state')
 need(set(b['rejections'])=={'mixed_unscheduled_row','wrong_known_mixed','wrong_slots_shape','wrong_native_epoch','negative_native_budget','submitted_over_budget','stale_tick_replay'} and not a['rejections'],'missing rejected native calls')
 need(all(aw['all_done']['status'][k]==0 for k in ['active','queued','tracked','inflight']),'terminal requests remain')
 return dict(native_decode_calls=native_calls,active_decode_occupancies=sorted(active_counts),mixed_ticks=mixed,full_four_row_state_digest_pairs=10*260,rejected_call_state_digest_pairs=520,exact_output_ids=sum(len(j['generated']) for j in a['jobs']))
