"""Audit native scheduled target spans and current ordinary complete state."""
from pathlib import Path
import copy,hashlib,json,sys
R=Path(__file__).resolve().parent;B=R.parent/'mistral-bounded-draft-long1'
sys.path.insert(0,str(B));from audit_qualification1 import check_input,need
sys.path.insert(0,str(R));from audit_common1 import validate
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
REFERENCE=read(B/'qualification1-result.json')['records'][0]
def prefill(ticks):
 need(len(ticks)==16,'missing native prefill chunk')
 for index,t in enumerate(ticks):
  p=t['plan'];need(p['stage']=='prefill' and len(p['requests'])==1 and p['epochs'][0]>0,'wrong native prefill plan');q=p['requests'][0]
  need(q['row']==0 and q['prefill_offset']==index*128 and q['prefill_tokens']==128 and q['cached_tokens']==0 and t['token_count']==(index+1)*128 and t['final']==(index==15),'native prefill progress differs')
 return ticks[0]['plan']['requests'][0]
def check(d):
 need(d['completed'] and d['experiment']=='qualify' and len(d['records'])==1 and d['split_scores_enabled'] and not d['fused_decode_enabled'],'wrong native mode')
 need(d['library']==str(R/'build-on1/libgarnet.dylib') and 'native_scheduled_dense_verify_v1' in d['backend'],'wrong native bridge')
 inp=copy.deepcopy(d['inputs']);tok=inp.pop('target_tokenizer_sha256');check_input(inp);need(tok==sha(Path(inp['target_model'])/'tokenizer.json'),'wrong native target tokenizer identity')
 row=d['records'][0];need(row['native_output_budget']==129,'state-witness headroom changed')
 summary=validate(d,dict(records=[dict(REFERENCE,mode='control')]),draft_family='qwen2')[0]
 for k in ['prompt_ids','generated_ids','slots','state_sha256','hidden_sha256']:need(row[k]==REFERENCE[k],'complete current ordinary mismatch '+k)
 first=prefill(row['prefill_ticks']);identity,epoch=first['id'],first['epoch'];left=128
 for step in row['rounds']:
  p=step['native_schedule'];need(p['stage']=='decode' and len(p['requests'])==1,'not one native scheduled decode');planned=p['requests'][0];q=step['native_commit'];n=step['consumed'];count=step['committed_before'];slot=2048+count-1
  need(planned['id']==q['id']==identity and planned['row']==q['row']==0 and planned['epoch']==q['epoch']==epoch and p['epochs']==[epoch] and p['active_rows']==[1],'row ownership/epoch differs')
  need(planned['token_id']==q['known']==row['generated_ids'][count-1] and planned['generated_tokens']==count and planned['slot']==q['slot_before']==slot,'native known/cursor differs')
  need(q['remaining_before']==129-count and q['submitted']==len(step['proposal'])+1 and q['consumed']==n and q['emitted']==step['emitted_ids'] and not q['finished'] and not q['eos'],'native budget/accepted span differs')
  metrics=step['metrics'];need(metrics['target_attention_split_score_dispatches']==32 and metrics['consumed_inputs']==[n] and metrics['emitted_tokens']==[q['emitted']] and metrics['eos']==[False],'native verifier/commit split differs')
 need(row['after_cancel']['active']==row['after_cancel']['queued']==row['after_cancel']['inflight']==row['after_cancel']['tracked']==0,'cancelled native row remains')
 terminal=row['terminal_budget2'];f2=prefill(terminal['prefill_ticks']);q=terminal['commit']['requests'][0]
 need(f2['id']>identity and f2['epoch']>epoch and q['id']==f2['id'] and q['epoch']==f2['epoch'] and q['row']==0 and q['known']==row['generated_ids'][0] and q['slot_before']==2048 and q['remaining_before']==q['submitted']==q['consumed']==1 and q['emitted']==row['generated_ids'][1:2] and q['finished'] and not q['eos'] and terminal['epoch_released'],'actual budget2 terminal/reuse differs')
 need(terminal['status']['active']==terminal['status']['queued']==terminal['status']['tracked']==terminal['status']['inflight']==0,'terminal request remains')
 expected={'no_scheduled_tick','wrong_known','wrong_slot','nonprefix_mask','no_active_scheduled_row','nonfinite_mask','wrong_position_axis','negative_token','wrong_token_count','wrong_mask_shape','depth_exceeds_native_budget','actual_terminal_budget_overflow'}
 need(set(row['rejections'])==expected and len(row['rejections'])==len(expected),'missing native rejection execution')
 return summary

def main():
 need(not (R/'qualification-audit1.json').exists(),'audit overwrite');need(read(R/'qualification1-status.json')['returncode']==0 and read(R/'qualification1-postflight.json')['passed'],'unfinished qualification')
 for n,h in read(R/'qualification1-preflight.json')['files_sha256'].items():need(sha(Path(n))==h,'admitted file changed '+n)
 d=read(R/'qualification1-result.json');summary=check(d);rejected=[]
 mutations={
 'wrong_output':lambda x:x['records'][0]['generated_ids'].__setitem__(0,-1),
 'wrong_kv':lambda x:x['records'][0]['state_sha256'][0].__setitem__(0,'0'*64),
 'wrong_hidden':lambda x:x['records'][0].update(hidden_sha256='0'*64),
 'fake_native_budget':lambda x:x['records'][0]['rounds'][0]['native_commit'].update(remaining_before=128),
 'wrong_epoch':lambda x:x['records'][0]['rounds'][0]['native_commit'].update(epoch=999),
 'wrong_native_known':lambda x:x['records'][0]['rounds'][0]['native_commit'].update(known=-1),
 'fake_split_counter':lambda x:x['records'][0]['rounds'][0]['metrics'].update(target_attention_split_score_dispatches=0),
 'missing_rollback':lambda x:next(q['metrics'] for q in x['records'][0]['rounds'] if q['consumed']<len(q['proposal'])+1).update(commit_kv_restores=0),
 'fake_terminal_release':lambda x:x['records'][0]['terminal_budget2'].update(epoch_released=False),
 'terminal_active':lambda x:x['records'][0]['terminal_budget2']['status'].update(active=1),
 'missing_native_rejection':lambda x:x['records'][0]['rejections'].pop(),
 'future_prefix':lambda x:next(q['draft'] for q in x['records'][0]['rounds'] if q['draft'].get('prompt_ids'))['prompt_ids'].append(1),
 }
 for name,fn in mutations.items():
  bad=copy.deepcopy(d);fn(bad)
  try:check(bad)
  except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
  else:raise ValueError('corruption admitted '+name)
 report=dict(passed=True,scope=__doc__,summary=summary,output_ids_exact_current_ordinary=True,full_kv_hidden_digest_pairs=65,target_prompt_tokens=2048,output_cap=128,native_witness_budget=129,witness_captured_before_cancel=True,actual_budget2_terminal_release_proved=True,native_api_rejections=len(d['records'][0]['rejections']),negative_controls_rejected=rejected,native_verification_calls=len(d['records'][0]['rounds']),split_layer_dispatches=32*len(d['records'][0]['rounds']),files_sha256={str(p):sha(p) for p in [R/'audit_qualification1.py',R/'audit_common1.py',R/'qualification1-result.json',R/'qualification1-preflight.json']},performance_measured=False,serving_measured=False,no_comparator_execution=True,goal_complete=False)
 (R/'qualification-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
