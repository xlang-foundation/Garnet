"""Audit B4 helper ownership, causal inputs and repeatability; no target/speed claim."""
from pathlib import Path
import copy,hashlib,json,math,re
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
def require(ok,why):
 if not ok:raise ValueError(why)
def rowstate(s,row):return [x[row] for x in s['kv']],None if s['hidden'] is None else s['hidden'][row]
def state(s,empty_hidden=False):
 require(set(s)=={'kv','hidden'} and len(s['kv'])==24,'state shape')
 for layer in s['kv']:
  require(len(layer)==4 and all(len(row)==2 for row in layer),'KV row shape')
  require(all(re.fullmatch('[0-9a-f]{64}',d) for row in layer for d in row),'KV digest')
 if s['hidden'] is None:require(empty_hidden,'missing initialized hidden bank')
 else:require(len(s['hidden'])==4 and all(re.fullmatch('[0-9a-f]{64}',d) for d in s['hidden']),'hidden shape/digest')
def flattened(f,key):
 out={}
 for result in f['results']:
  for label,v in result[key].items():require(label not in out,'duplicate result owner');out[label]=v
 return out
def audit_run(d,mode,inputs):
 require(d['completed'] and d['mode']==mode and d['inputs']==inputs,'run completion/input binding')
 require(d['library']==str(R/'build-on1/libgarnet.dylib') and 'native_mistral_verification_gqa16_values64_32_optional_v1' in d['backend'],'qualified native library')
 require(len(d['frames'])==len(inputs['frames'])==5,'frame count')
 require(len(d['claims'])==5 and [c['label'] for c in d['claims']]==['A','B','C','D','E'],'claim sequence')
 claims={c['label']:c for c in d['claims']}
 require([claims[x]['row'] for x in inputs['initial']]==list(range(4)),'initial row mapping')
 require(claims['E']['row']==claims['B']['row'] and claims['E']['epoch']>claims['B']['epoch']>0,'replacement epoch')
 require([c['request'] for c in d['claims']]==[1,2,3,4,5],'request identity')
 require(d['rejections']==[dict(kind='stale_replaced_epoch',frame=2,row=1)],'stale epoch rejection')
 require(d['prepared']['prefill_graphs']==24,'prepared layer profile')
 memory=d['prepared']['memory'];require(memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-536870912,'helper memory bound')
 cleanup=d['cleanup'];require(all(cleanup[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')) and cleanup['metal_allocated_bytes']<=4194304,'cleanup')
 histories={x:[] for x in inputs['initial']};leases={x:{k:v for k,v in claims[x].items() if k!='label'} for x in inputs['initial']}
 expected_calls=[];reconcile_calls=0;survivor_pairs=0;zero_bytes=0;replays=0;draft_tokens=0;previous=None
 for index,(fixture,f) in enumerate(zip(inputs['frames'],d['frames'])):
  require(f['index']==index,'frame order')
  for phase in ('before','after_reconcile','after'):state(f[phase],index==0 and phase!='after')
  if previous is not None:require(f['before']==previous,'state handoff')
  released=None
  if fixture.get('release'):
   label=fixture['release'];released=leases.pop(label);require(f['released']==released,'released lease');histories.pop(label)
   new=fixture['claim'];leases[new]={k:v for k,v in claims[new].items() if k!='label'};histories[new]=[]
  else:require(f['released'] is None,'unexpected release')
  require(f['leases']==leases,'live lease ownership')
  queries=fixture['queries'];labels=[q['label'] for q in queries];require(len(set(labels))==len(labels),'duplicate query')
  require(len(f['reconcile'])==len(queries),'reconcile count');offset={}
  for q,c in zip(queries,f['reconcile']):
   label=q['label'];old=histories[label];ids=q['prompt_ids'];common=0
   while common<min(len(old),len(ids)) and old[common]==ids[common]:common+=1
   kept=common-(common==len(ids) and common>0 and common<len(old));replay=kept<common
   require(c['label']==label and {k:c[k] for k in ('row','epoch','request')}==leases[label],'reconcile owner')
   reconcile_calls+=1;s=c['stats'];expected_bytes=(len(old)-kept)*24576+3584 if kept<len(old) else 0
   expected=dict(calls=reconcile_calls,previous_tokens=len(old),common_tokens=common,retained_tokens=kept,zeroed_bytes=expected_bytes,replay_last_token=replay,host_snapshot_bytes=0)
   require(set(s)==set(expected)|{'gpu_ms'} and all(s[k]==v for k,v in expected.items()),'reconcile prefix/zeroing')
   require(math.isfinite(s['gpu_ms']) and s['gpu_ms']>=0,'reconcile GPU time')
   offset[label]=kept;zero_bytes+=expected_bytes;replays+=replay
   if kept==len(old) and not (released and released['row']==leases[label]['row']):require(rowstate(f['before'],leases[label]['row'])==rowstate(f['after_reconcile'],leases[label]['row']),'unchanged retained row')
  touched={leases[x]['row'] for x in labels};reconcile_touched=touched|({released['row']} if released else set())
  for row in range(4):
   if row not in reconcile_touched:
    require(rowstate(f['before'],row)==rowstate(f['after_reconcile'],row),'reconcile changed survivor');survivor_pairs+=48+(f['before']['hidden'] is not None)
   if row not in touched:
    require(rowstate(f['after_reconcile'],row)==rowstate(f['after'],row),'execution changed survivor');survivor_pairs+=48+(f['after_reconcile']['hidden'] is not None)
  if released:
   row=released['row'];require([l[row] for l in f['after_reconcile']['kv']]==[l[0] for l in d['frames'][0]['before']['kv']],'released row not cleared')
  values={k:flattened(f,k) for k in ('tokens','executed','stop')}
  require(all(set(v)==set(labels) for v in values.values()),'result owner set')
  for q in queries:
   label=q['label'];tokens=values['tokens'][label];executed=values['executed'][label];stop=values['stop'][label]
   require(all(type(x) is int and 0<=x<151936 and x not in inputs['draft_stops'] for x in tokens),'invalid draft token')
   require(stop in ('eos','limit') and 0<=len(tokens)<=q['draft_limit']<=8,'draft stop bounds')
   require((stop=='limit' and len(tokens)==q['draft_limit'] and executed==tokens[:-1]) or (stop=='eos' and len(tokens)<q['draft_limit'] and executed==tokens),'draft execution boundary')
   draft_tokens+=len(tokens);histories[label]=q['prompt_ids']+executed
  groups=[queries] if mode=='batch' else [[q] for q in queries]
  require(len(f['results'])==len(groups),'result group count')
  for group,result in zip(groups,f['results']):
   require(set(result)=={'tokens','executed','stop','prefill_calls'},'result schema')
   require(set(result['tokens'])=={q['label'] for q in group},'result grouping')
   cursors=dict(offset);prefill_indices=[]
   while any(cursors[q['label']]<len(q['prompt_ids']) for q in group):
    byrow={leases[q['label']]['row']:q for q in group};ids=[];slots=[];active=[];selected=[];lengths={}
    for row in range(4):
     q=byrow.get(row);n=min(16,len(q['prompt_ids'])-cursors[q['label']]) if q else 0
     if n:selected.append(q['label']);lengths[q['label']]=n
     for t in range(16):
      on=t<n;position=cursors[q['label']]+t if on else 0
      ids.append(q['prompt_ids'][position] if on else 0);slots.append(float(position));active.append(float(on))
    epochs=[0]*4
    for x in selected:epochs[leases[x]['row']]=leases[x]['epoch']
    prefill_indices.append(len(expected_calls));expected_calls.append(dict(stage='prefill',labels=selected,epochs=epochs,tokens=ids,slots=slots,active=active,offsets=dict(cursors),lengths=lengths))
    for x,n in lengths.items():cursors[x]+=n
   require(result['prefill_calls']==prefill_indices,'prefill call indices')
   for step in range(max((len(values['executed'][q['label']]) for q in group),default=0)):
    ids=[0]*4;slots=[0.]*4;active=[0.]*4;epochs=[0]*4;advancing=[]
    for q in group:
     label=q['label'];executed=values['executed'][label]
     if step>=len(executed):continue
     row=leases[label]['row'];advancing.append(label);ids[row]=executed[step];slots[row]=float(len(q['prompt_ids'])+step);active[row]=1.;epochs[row]=leases[label]['epoch']
    expected_calls.append(dict(stage='decode',labels=advancing,epochs=epochs,tokens=ids,slots=slots,active=active))
  previous=f['after']
 require(d['calls']==expected_calls,'native call masks/tokens/slots/epochs')
 return dict(frames=5,queries=reconcile_calls,draft_tokens=draft_tokens,native_calls=len(expected_calls),survivor_digest_pairs=survivor_pairs,reconcile_zeroed_bytes=zero_bytes,replay_last_token_cases=replays)
def audit(a,b,c,inputs):
 runs=[audit_run(d,m,inputs) for d,m in [(a,'serial'),(b,'batch'),(c,'batch')]]
 require(a['claims']==b['claims']==c['claims'],'cross-run ownership')
 paired=0;serial_different=0
 for x,y,z in zip(a['frames'],b['frames'],c['frames']):
  for k in ('tokens','executed','stop'):require(flattened(x,k)==flattened(y,k)==flattened(z,k),'cross-run helper outputs')
  for phase in ('before','after_reconcile','after'):
   require(y[phase]==z[phase],'batch repeat full-state mismatch')
   paired+=192+(4 if y[phase]['hidden'] is not None else 0)
   serial_different+=sum(x[phase]['kv'][l][r][p]!=y[phase]['kv'][l][r][p] for l in range(24) for r in range(4) for p in range(2))
   if x[phase]['hidden'] is not None:serial_different+=sum(u!=v for u,v in zip(x[phase]['hidden'],y[phase]['hidden']))
 require(b['calls']==c['calls'],'batch repeat call trace')
 return dict(passed=True,runs=runs,batch_repeat_full_state_digest_pairs=paired,serial_vs_batch_different_state_digests=serial_different,serial_vs_batch_helper_token_ids_equal=True,serial_vs_batch_full_state_bit_equal=serial_different==0,helper_only=True,productive_target_batch_qualified=False,performance_claim=False,no_comparator_execution=True,goal_complete=False)
def main():
 inputs=read('input1.json');a=read('serial-result2.json');b=read('batch-result2.json');c=read('batch-repeat-result3.json')
 report=audit(a,b,c,inputs);negatives=[]
 def reject(name,mutate):
  v=copy.deepcopy(b);mutate(v)
  try:audit(a,v,c,inputs)
  except (ValueError,KeyError,IndexError,TypeError):negatives.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('draft token',lambda v:v['frames'][0]['results'][0]['tokens']['A'].__setitem__(0,1))
 reject('wrong owner epoch',lambda v:v['frames'][1]['reconcile'][0].__setitem__('epoch',0))
 reject('wrong retained prefix',lambda v:v['frames'][1]['reconcile'][0]['stats'].__setitem__('retained_tokens',48))
 reject('wrong zeroed bytes',lambda v:v['frames'][1]['reconcile'][0]['stats'].__setitem__('zeroed_bytes',0))
 reject('host prefix snapshot',lambda v:v['frames'][1]['reconcile'][0]['stats'].__setitem__('host_snapshot_bytes',1))
 reject('missing stale rejection',lambda v:v.__setitem__('rejections',[]))
 reject('survivor KV overwrite',lambda v:v['frames'][1]['after']['kv'][0][1].__setitem__(0,'0'*64))
 reject('survivor hidden overwrite',lambda v:v['frames'][1]['after']['hidden'].__setitem__(1,'0'*64))
 reject('native slot',lambda v:v['calls'][0]['slots'].__setitem__(0,1.))
 reject('native inactive mask',lambda v:v['calls'][0]['active'].__setitem__(0,0.))
 reject('native token',lambda v:v['calls'][0]['tokens'].__setitem__(0,1))
 reject('missing native call',lambda v:v['calls'].pop())
 reject('replacement epoch alias',lambda v:v['claims'][4].__setitem__('epoch',v['claims'][1]['epoch']))
 reject('lost final-token replay',lambda v:v['frames'][4]['reconcile'][0]['stats'].__setitem__('replay_last_token',False))
 reject('owner leak',lambda v:v['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 reject('foreign native library',lambda v:v.__setitem__('library','/tmp/foreign.dylib'))
 report['negative_controls_rejected']=negatives
 report['source_interpretation']='The existing FP32 chunk-attention policy selects scalar attention at at most eight active tokens across the batch. This can change floating-point state when a companion request changes that count; serial/batch state equality is explicitly not admitted. Identical-batch repeat state and all serial/batch helper token IDs passed this fixture.'
 report['files_sha256']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['input1.json','serial-result2.json','batch-result2.json','batch-repeat-result3.json','helper_worker2.py','run_helper2.py','run_helper3.py','helper2-preflight.json','helper2-postflight.json','helper3-preflight.json','helper3-postflight.json','build-audit1.json','source-plan1.json']}
 (R/'helper-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
