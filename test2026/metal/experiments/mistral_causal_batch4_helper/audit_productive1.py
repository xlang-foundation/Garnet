"""Independent CPU audit of causal B4 drafts and ordinary target state equivalence."""
from pathlib import Path
import copy,hashlib,json,math,re
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent
read=lambda n:json.loads((R/n).read_text())
def need(ok,why):
 if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def kvrow(states,row):return [layer[row] for layer in states]
def checkstates(states,layers):
 need(len(states)==layers,'state layer count')
 for layer in states:
  need(len(layer)==4 and all(len(row)==2 for row in layer),'state row shape')
  need(all(re.fullmatch('[0-9a-f]{64}',d) for row in layer for d in row),'state digest')
def target_run(d,inputs,mode):
 need(d['completed'] and d['mode']==mode and d['inputs']==inputs,'run binding/completion')
 need(d['library']==str(R/'build-on1/libgarnet.dylib') and 'native_mistral_verification_gqa16_values64_32_optional_v1' in d['backend'],'qualified native library')
 need(len(d['jobs'])==4 and [j['label'] for j in d['jobs']]==list('ABCD'),'target job set')
 ids={j['native_request']:j['label'] for j in d['jobs']};fixtures={j['label']:j for j in inputs['jobs']};hist={x:[] for x in ids.values()};prefilled={x:0 for x in ids.values()};placements={};finished=set();si=0;decodes=[];states={}
 for j in d['jobs']:
  need(all(j[k]==fixtures[j['label']][k] for k in fixtures[j['label']]),'target fixture change')
  need(j['finished'] and len(j['generated'])<=j['limit'],'target finish bounds')
 for tick in d['ticks']:
  need(len(tick['plan']['batches'])==len(tick['executions']),'plan/execution count')
  for plan,event in zip(tick['plan']['batches'],tick['executions']):
   need(plan['stage']==event['stage'],'stage mismatch')
   for q in plan['requests']:
    label=ids[q['id']];need(label not in finished,'finished target rescheduled')
    lease=(q['row'],q['epoch']);need(0<=lease[0]<4 and lease[1]>0,'target lease bounds')
    need(placements.get(label,lease)==lease,'target placement changed');placements[label]=lease
   completed={c['id']:c for c in event['completed']}
   if plan['stage']=='prefill':
    for q in plan['requests']:
     label=ids[q['id']];f=fixtures[label];need(q['prefill_offset']==prefilled[label] and q['prefill_tokens']>0,'prefill progression');prefilled[label]+=q['prefill_tokens'];need(prefilled[label]<=len(f['prompt_ids']),'prefill overrun')
     if prefilled[label]==len(f['prompt_ids']):
      need(q['id'] in completed,'missing first prediction');c=completed.pop(q['id']);hist[label].append(c['token']);need(c['finished']==(c['token'] in inputs['stops'] or len(hist[label])==f['limit']),'prefill terminal flag')
      if c['finished']:finished.add(label)
    need(not completed,'foreign prefill completion')
   else:
    need(plan['stage']=='decode' and set(completed)=={q['id'] for q in plan['requests']},'decode request set')
    before={ids[q['id']]:list(hist[ids[q['id']]]) for q in plan['requests']}
    for q in plan['requests']:
     label=ids[q['id']];f=fixtures[label];c=completed[q['id']];need(q['token_id']==hist[label][-1] and q['slot']==len(f['prompt_ids'])+len(hist[label])-1,'known token/slot')
     need((c['row'],c['epoch'])==placements[label],'completion owner')
     emitted=c['emitted'];need(emitted and all(type(t) is int and 0<=t<32768 for t in emitted),'target emitted IDs')
     if mode=='ordinary':need(len(emitted)==1 and event['native'] is None,'ordinary event')
     else:
      proposal=event['proposals'][str(q['id'])];need(len(proposal)<=min(3,f['limit']-len(hist[label])-1),'proposal output budget')
      need(c['known']==q['token_id'] and c['slot_before']==q['slot'] and c['remaining_before']==f['limit']-len(hist[label]) and c['submitted']==len(proposal)+1,'native verification schedule binding')
      need(c['consumed']==len(emitted) and 1<=len(emitted)<=len(proposal)+1 and emitted[:-1]==proposal[:len(emitted)-1],'accepted target span')
     hist[label].extend(emitted);need(len(hist[label])<=f['limit'],'target output limit');terminal=emitted[-1] in inputs['stops'] or len(hist[label])==f['limit'];need(c['finished']==terminal,'target finish flag')
     if terminal:finished.add(label)
    if mode=='productive':
     native=event['native'];need(native['requests']==event['completed'],'native completion publication');m=native['verification']
     need(m['target_affine_misses']==0 and m['target_indexed_decode_dispatches']==225 and m['target_mpp_affine_dispatches']==m['target_indexed_prefill_dispatches']==0,'target FP32 projection route')
     need(m['target_attention_gqa16_score_dispatches']==m['target_attention_gqa16_value_dispatches']==32 and m['target_attention_vector4_dispatches']==0,'qualified shared-KV route')
     byrow={q['row']:q for q in plan['requests']};flat=[];slots=[];active=[]
     for row in range(4):
      q=byrow.get(row);tokens=[q['token_id']]+event['proposals'][str(q['id'])] if q else []
      for t in range(4):on=t<len(tokens);flat.append(tokens[t] if on else 0);slots.append(float(q['slot']+t) if on else 0.);active.append(float(on))
     need(event['submitted_ids']==flat and event['submitted_slots']==slots and event['submitted_active']==active,'submitted target tokens/slots/masks')
    decodes.append(dict(plan=plan,event=event,before=before))
   snap=d['snapshots'][si];si+=1;need(snap['stage']==plan['stage'],'snapshot stage');checkstates(snap['states'],32);need(len(snap['hidden'])==4 and all(re.fullmatch('[0-9a-f]{64}',h) for h in snap['hidden']),'target hidden digests')
   owners={o['label']:o for o in snap['owners']};need(set(owners)==set(placements),'snapshot owner set')
   for label,o in owners.items():
    need((o['row'],o['epoch'])==placements[label] and o['generated_ids']==hist[label] and o['prefilled']==prefilled[label] and o['finished']==(label in finished),'snapshot progress')
    key=(label,prefilled[label],len(hist[label]),label in finished);value=(kvrow(snap['states'],o['row']),snap['hidden'][o['row']],list(hist[label]))
    need(key not in states or states[key]==value,'unchanged target owner state');states[key]=value
 need(si==len(d['snapshots']) and len(finished)==4,'target snapshots/completion')
 need(all(j['generated']==hist[j['label']] for j in d['jobs']),'published final output')
 cleanup=d['cleanup'];need(all(cleanup[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')) and cleanup['metal_allocated_bytes']<=4194304,'target cleanup')
 need(d['prepared']['memory']['metal_allocated_bytes']<=d['prepared']['memory']['metal_recommended_working_set_bytes']-536870912,'target memory bound')
 return dict(states=states,decodes=decodes,histories=hist)
def audit(ordinary,candidate,inputs,ttok,dtok):
 a=target_run(ordinary,inputs,'ordinary');b=target_run(candidate,inputs,'productive');need(a['histories']==b['histories'],'ordinary/verified target token mismatch')
 state_pairs=0
 for key,value in b['states'].items():need(key in a['states'] and a['states'][key]==value,'ordinary/verified target KV or hidden mismatch');state_pairs+=65
 need(len(candidate['helper_rounds'])==len(b['decodes']),'helper round count')
 fixtures={j['label']:j for j in inputs['jobs']};histories={x:[] for x in fixtures};epochs={};expected_calls=[];reconcile_count=0;survivor_pairs=0;draft_count=0;accepted=0;proposed=0;previous=None;live=set(fixtures);zeroed=0
 for event,round in zip(b['decodes'],candidate['helper_rounds']):
  before_target=event['before'];traces=round['traces'];need({t['label'] for t in traces}==set(before_target) and len(traces)==len(before_target),'helper selected owner set')
  for phase in ('before','after_reconcile','after'):checkstates(round[phase],24)
  if previous is not None:
   for label in live:row='ABCD'.index(label);need(kvrow(round['before'],row)==kvrow(previous,row),'helper survivor handoff')
  selected=[];offset={};bylabel={t['label']:t for t in traces}
  for q in event['plan']['requests']:
   label=next(j['label'] for j in candidate['jobs'] if j['native_request']==q['id']);t=bylabel[label];generated=before_target[label];remaining=fixtures[label]['limit']-len(generated);limit=min(3,remaining-1);row='ABCD'.index(label)
   need(t['target_request']==q['id'] and t['target_row']==q['row'] and t['target_epoch']==q['epoch'] and t['generated_before']==generated and t['remaining']==remaining and t['target_proposal_limit']==limit and t['helper_row']==row,'causal helper owner/input binding')
   need(t['helper_epoch']>0 and epochs.get(label,t['helper_epoch'])==t['helper_epoch'],'helper epoch');epochs[label]=t['helper_epoch']
   need(t['proposal']==event['event']['proposals'][str(q['id'])],'helper/target proposal binding')
   prefix=ttok.decode(generated,skip_special_tokens=False);canonical=ttok.encode(prefix,add_special_tokens=False).ids==generated
   prompt=dtok.encode(fixtures[label]['draft_prompt']+prefix,add_special_tokens=False).ids
   bypass='clipped_tail' if limit==0 else 'noncanonical_prefix' if not canonical else 'draft_capacity' if len(prompt)+8>512 else None
   if bypass:
    need(t['reason']==bypass and t['prompt_ids'] is None and t['reconcile'] is None and not t['proposal'] and not t['draft_ids'] and not t['executed'],'helper admission bypass');continue
   need(t['prompt_ids']==prompt and t['prefix']==prefix,'helper sees only committed target prefix')
   old=histories[label];common=0
   while common<min(len(old),len(prompt)) and old[common]==prompt[common]:common+=1
   kept=common-(common==len(prompt) and common>0 and common<len(old));reconcile_count+=1;s=t['reconcile'];bytes=(len(old)-kept)*24576+3584 if kept<len(old) else 0
   expected=dict(calls=reconcile_count,previous_tokens=len(old),common_tokens=common,retained_tokens=kept,zeroed_bytes=bytes,replay_last_token=kept<common,host_snapshot_bytes=0)
   need(set(s)==set(expected)|{'gpu_ms'} and all(s[k]==v for k,v in expected.items()) and math.isfinite(s['gpu_ms']) and s['gpu_ms']>=0,'helper reconciliation')
   tokens=t['draft_ids'];executed=t['executed'];need(len(tokens)<=8 and all(type(x) is int and 0<=x<151936 and x not in inputs['draft_stops'] for x in tokens),'helper draft IDs')
   text=dtok.decode(tokens,skip_special_tokens=False);translated=ttok.encode(prefix+text,add_special_tokens=False).ids if tokens else []
   proposal=translated[len(generated):len(generated)+limit] if tokens and translated[:len(generated)]==generated else []
   need(t['translated_ids']==translated and t['proposal']==proposal,'cross-tokenizer translation')
   reason=t['reason'];need(reason in ('draft_eos','target_limit','draft_budget'),'helper stop reason')
   if reason=='target_limit':need(len(proposal)==limit and executed==tokens[:-1],'helper proposal stop')
   elif reason=='draft_eos':need(executed==tokens and len(tokens)<8,'helper EOS stop')
   else:need(len(tokens)==8 and executed==tokens[:-1],'helper budget stop')
   histories[label]=prompt+executed;offset[label]=kept;selected.append(label);draft_count+=len(tokens);zeroed+=bytes
  touched={'ABCD'.index(x) for x in selected}
  for row in range(4):
   if row not in touched:
    need(kvrow(round['before'],row)==kvrow(round['after_reconcile'],row)==kvrow(round['after'],row),'helper untouched row changed');survivor_pairs+=96
  need(round['call_first']==len(expected_calls),'helper call range start');cursors=dict(offset)
  while any(cursors[x]<len(bylabel[x]['prompt_ids']) for x in selected):
   ids=[];slots=[];active=[];labels=[];lengths={};ep=[0]*4
   for row in range(4):
    label='ABCD'[row];t=bylabel.get(label);length=min(16,len(t['prompt_ids'])-cursors[label]) if label in selected else 0
    if length:labels.append(label);lengths[label]=length;ep[row]=epochs[label]
    for j in range(16):on=j<length;pos=cursors[label]+j if on else 0;ids.append(t['prompt_ids'][pos] if on else 0);slots.append(float(pos));active.append(float(on))
   expected_calls.append(dict(stage='prefill',labels=labels,tokens=ids,slots=slots,active=active,epochs=ep))
   for label,n in lengths.items():cursors[label]+=n
  for step in range(max((len(bylabel[x]['executed']) for x in selected),default=0)):
   ids=[0]*4;slots=[0.]*4;active=[0.]*4;ep=[0]*4;labels=[]
   for t in traces:
    label=t['label'];row='ABCD'.index(label)
    if label not in selected or step>=len(t['executed']):continue
    labels.append(label);ids[row]=t['executed'][step];slots[row]=float(len(t['prompt_ids'])+step);active[row]=1.;ep[row]=epochs[label]
   expected_calls.append(dict(stage='decode',labels=labels,tokens=ids,slots=slots,active=active,epochs=ep))
  need(round['call_last']==len(expected_calls),'helper call range end');previous=round['after']
  for c in event['event']['completed']:
   proposed+=c['submitted']-1;accepted+=c['consumed']-1
   if c['finished']:
    label=next(j['label'] for j in candidate['jobs'] if j['native_request']==c['id']);live.remove(label);histories[label]=[]
 need(candidate['helper_calls']==expected_calls,'helper native tokens/slots/masks/epochs')
 need(proposed>0 and accepted>0,'no productive helper acceptance')
 return dict(passed=True,exact_target_output_ids=sum(map(len,b['histories'].values())),ordinary_decode_calls=len(a['decodes']),productive_decode_calls=len(b['decodes']),accepted_draft_tokens=accepted,proposed_target_tokens=proposed,helper_draft_tokens=draft_count,helper_native_calls=len(expected_calls),helper_reconciliation_calls=reconcile_count,helper_zeroed_bytes=zeroed,helper_unchanged_digest_pairs=survivor_pairs,ordinary_vs_verified_target_digest_pairs=state_pairs,reference_outputs_read_during_generation=False,performance_claim=False,http_qualified=False,quality_suite_qualified=False,no_comparator_execution=True,goal_complete=False)
def main():
 inputs=read('productive-input1.json');ordinary=read('ordinary-target-result1.json');candidate=read('productive-target-result1.json');ttok=Tokenizer.from_file(inputs['target_model']+'/tokenizer.json');dtok=Tokenizer.from_file(inputs['draft_model']+'/tokenizer.json');report=audit(ordinary,candidate,inputs,ttok,dtok);negative=[]
 def reject(name,mutate):
  bad=copy.deepcopy(candidate);mutate(bad)
  try:audit(ordinary,bad,inputs,ttok,dtok)
  except (ValueError,KeyError,TypeError,IndexError,StopIteration):negative.append(name);return
  raise AssertionError('accepted corruption '+name)
 reject('published target output',lambda d:d['jobs'][0]['generated'].__setitem__(0,1))
 reject('target hidden state',lambda d:d['snapshots'][-1]['hidden'].__setitem__(0,'0'*64))
 reject('target KV state',lambda d:d['snapshots'][-1]['states'][0][0].__setitem__(0,'0'*64))
 reject('future prefix injection',lambda d:d['helper_rounds'][0]['traces'][0]['generated_before'].append(1))
 reject('wrong helper owner epoch',lambda d:d['helper_rounds'][0]['traces'][0].__setitem__('helper_epoch',0))
 reject('wrong helper prompt',lambda d:d['helper_rounds'][0]['traces'][0]['prompt_ids'].__setitem__(0,0))
 reject('host snapshot insertion',lambda d:d['helper_rounds'][0]['traces'][0]['reconcile'].__setitem__('host_snapshot_bytes',1))
 reject('helper inactive mask',lambda d:d['helper_calls'][0]['active'].__setitem__(0,0.))
 reject('helper execution slot',lambda d:d['helper_calls'][0]['slots'].__setitem__(0,1.))
 reject('missing helper call',lambda d:d['helper_calls'].pop())
 reject('helper output leak',lambda d:d['cleanup'].__setitem__('garnet_live_matrix_owners',1))
 reject('foreign native library',lambda d:d.__setitem__('library','/tmp/foreign.dylib'))
 report['negative_controls_rejected']=negative;report['files_sha256']={n:sha(R/n) for n in ['productive-input1.json','productive_worker1.py','run_productive1.py','ordinary-target-result1.json','productive-target-result1.json','productive1-preflight.json','productive1-postflight.json','build-audit1.json','helper-audit1.json']}
 (R/'productive-audit1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'},indent=2))
if __name__=='__main__':main()
