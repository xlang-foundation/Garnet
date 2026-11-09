"""Audit prepared compact/verification coexistence, fixed profiles and lifecycle."""
from pathlib import Path
import copy,hashlib,json,sys
R=Path(__file__).resolve().parent;M=R.parent
sys.path.insert(0,str(R));from audit_core1 import check,need
read=lambda p:json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 need(not (R/'qualification-audit1.json').exists(),'audit overwrite');pre=read(R/'qualification1-preflight.json');need(pre['passed'] and read(R/'qualification1-postflight.json')['passed'],'unfinished collection')
 for n,h in pre['files_sha256'].items():need(sha(Path(n))==h,'admitted file changed '+n)
 previous=0;pids=set();guards=[];guard_pairs=0
 for label,worker,build,mode in read(R/'qualification1-sequence.json'):
  s=read(R/(label+'-status1.json'));need(s['status']=='terminal' and s['returncode']==0 and s['pid'] not in pids and s['finished_ns']>s['started_ns']>=previous,'not serial/fresh/terminal');previous=s['finished_ns'];pids.add(s['pid'])
  if not label.startswith('guard-'):continue
  d=read(R/(label+'-result1.json'));need(d['completed'] and d['mode']==mode,'guard incomplete');wanted=2 if mode=='on-owned' else 1;need(len(d['rejections'])==wanted,'guard coverage');end=d['cleanup'];need(end['metal_allocated_bytes']<=4194304 and all(end[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'guard owners retained')
  for q in d['rejections']:
   need(q['before']==q['after'] and q['error'] and len(q['before']['kv'])==32 and len(q['before']['hidden'])==4,'rejected preparation mutated state');guard_pairs+=260
  guards.append(dict(mode=mode,rejections=[q['label'] for q in d['rejections']]))
 a,b,c=(read(R/(mode+'-result1.json')) for mode in ['ordinary','verify','reverse'])
 need(c['mode']=='reverse','missing reverse preparation order');view=copy.deepcopy(c);view['mode']='verify';summary=check(a,b);reverse_summary=check(a,view)
 profiles=set();b1_routes=0
 for d in [a,b,c]:
  need('native_compact_verify_b4t1_optional_v1' in d['backend'] and d['library']==str(R/'build-on1/libgarnet.dylib'),'wrong enabled build')
  need(d['prepared']['compact_families']==17 and d['prepared']['maximum_compact_chunk']==512,'prepared up-to profile incomplete')
  local=set();compact=0
  for tick in d['ticks']:
   for p,e in zip(tick['plan']['batches'],tick['executions']):
    if p['stage']=='decode':need(e['compact_rows']==0,'verification/ordinary decode used compact prefill');continue
    width=e['physical_chunk'];need(width==p['prefill_chunk'] and width in [2,4,8,16,32,64,128,256,512],'unprepared chunk')
    actual=sum(q['prefill_tokens'] for q in p['requests']);need(actual+sum(len(x['requests']) for x in tick['plan']['batches'] if x['stage']=='decode')<=512,'global token budget exceeded')
    rows=[q['row'] for q in p['requests']];n=e['compact_rows'];need(n in [0,1,2],'bad compact row count')
    if n:need(len(rows)==n and rows==list(range(rows[0],rows[0]+n)) and e['compact_first']==rows[0],'compact physical span differs');compact+=1
    local.add((width,n))
  need({(128,0),(256,1),(512,1)}<=local,'actual up-to chunks not exercised');profiles|=local;b1_routes+=compact
 old=read(M/'mistral-scheduled-c4-ownership2/ordinary-result1.json');prior={x['label']:x for x in old['witnesses']};current={x['label']:x for x in a['witnesses']}
 for label in ['initial_prefilled','finish_A']:need(prior[label]['states']==current[label]['states'] and prior[label]['hidden']==current[label]['hidden'],'initial full-batch producer state changed')
 for j in old['jobs']:
  now=next(x for x in a['jobs'] if x['label']==j['label']);need(j['generated'][:len(now['generated'])]==now['generated'],'previous profile output prefix differs')
 # Repeat the lifecycle auditor with independent corruptions of the enabled arm.
 rejected=[]
 for name,fn in [
  ('survivor_kv',lambda d:next(x for x in d['witnesses'] if x['label']=='claim_E')['states'][0][1].__setitem__(0,'0'*64)),
  ('survivor_hidden',lambda d:next(x for x in d['witnesses'] if x['label']=='cancel_C')['hidden'].__setitem__(3,'0'*64)),
  ('wrong_epoch',lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')['native']['requests'][0].update(epoch=999)),
  ('fake_budget',lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='decode')['native']['requests'][0].update(remaining_before=999)),
  ('fake_compact_span',lambda d:next(e for t in d['ticks'] for e in t['executions'] if e['stage']=='prefill' and e['compact_rows']) .update(compact_first=3)),
  ('wrong_target_id',lambda d:d['jobs'][1]['generated'].__setitem__(0,-1)),
  ('live_owner',lambda d:d['cleanup'].update(garnet_live_matrix_owners=1))]:
  bad=copy.deepcopy(b);fn(bad)
  try:check(a,bad)
  except (ValueError,KeyError,TypeError,IndexError):rejected.append(name)
  else:raise ValueError('corruption admitted '+name)
 report=dict(passed=True,scope=__doc__,forward_summary=summary,reverse_summary=reverse_summary,full_four_row_state_digest_pairs=summary['full_four_row_state_digest_pairs']+reverse_summary['full_four_row_state_digest_pairs'],rejected_api_state_digest_pairs=520,preparation_guard_state_digest_pairs=guard_pairs,guards=guards,actual_profile_routes=[list(x) for x in sorted(profiles)],b1_compact_calls_all_three_arms=b1_routes,initial_full_batch_state_exact_prior_profile=True,all_full_state_and_outputs_exact_current_ordinary=True,preparation_orders=['compact_then_verification','verification_then_compact'],native_api_rejections_per_verifier_arm=3,negative_controls_rejected=rejected,files_sha256={str(p):sha(p) for p in [R/'audit_qualification1.py',R/'audit_core1.py',R/'ordinary-result1.json',R/'verify-result1.json',R/'reverse-result1.json',R/'qualification1-preflight.json']},b2_compact_interleave_qualified=False,c4t4_qualified=False,performance_measured=False,no_comparator_execution=True,goal_complete=False)
 (R/'qualification-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
