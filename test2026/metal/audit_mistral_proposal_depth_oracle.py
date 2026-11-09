#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent acceptance-only audit; cannot qualify native target8 or speed."""
import argparse,copy,hashlib,json,math
from pathlib import Path
from tokenizers import Tokenizer

def need(v,msg):
 if not v:raise ValueError(msg)
def goodtime(v):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and v>=0

def validate(data,baseline,adaptive=False):
 need(data['completed'] is True and len(data['records'])==(8 if adaptive else 16),'incomplete')
 cfg=data['inputs'];need(len(cfg['cases'])==8 and cfg['capacity']==4096 and cfg['draft_capacity']==512 and cfg['output_limit']==1024,'profile')
 need(data['prepared']['graphs']==112,'unprepared graph family')
 target=Tokenizer.from_file(cfg['target_model']+'/tokenizer.json');draft=Tokenizer.from_file(cfg['draft_model']+'/tokenizer.json')
 counter=0;summary=[]
 for index,case in enumerate(cfg['cases']):
  stride=1 if adaptive else 2;pair=data['records'][index*stride:index*stride+stride]
  need([r['depth'] for r in pair]==(['adaptive'] if adaptive else [3,7] if index%2==0 else [7,3]),'order')
  oracle=json.loads(Path(case['control']).read_text());need(oracle['prompt_ids']==case['prompt_ids'],'oracle prompt')
  for row in pair:
   need(row['case_id']==case['id'] and row['generated_ids']==oracle['generated_ids'],'case/output')
   policy=row['depth'];previous_accepted=0
   if adaptive:need(policy=='adaptive' and row['draft_budget']=='per-round','policy')
   else:depth=policy;budget=8 if depth==3 else 16;need(row['draft_budget']==budget,'budget')
   ids=row['generated_ids'];cursor=1;history=[]
   need(goodtime(row['draft_only_seconds']),'duration')
   for step in row['rounds']:
    if adaptive:
     depth=7 if previous_accepted>=3 else 3;budget=8 if depth==3 else 16
     need(step['depth']==depth and step['draft_budget']==budget,'noncausal adaptive depth')
    need(step['committed_before']==cursor,'future cursor');trace=step['draft'];proposal=step['proposal'];remaining=1024-cursor
    need(len(proposal)<=min(depth,remaining-1) and goodtime(trace['seconds']),'proposal bounds')
    prefix=target.decode(ids[:cursor],skip_special_tokens=False);canonical=target.encode(prefix,add_special_tokens=False).ids==ids[:cursor]
    prompt=draft.encode(case['draft_prompt']+prefix,add_special_tokens=False).ids
    reason=trace['reason']
    if reason in ('clipped_tail','noncanonical_prefix','draft_capacity'):
     need(not proposal,'fallback proposal')
     if reason=='clipped_tail':need(remaining==1,'tail')
     elif reason=='noncanonical_prefix':need(not canonical,'canonical fallback')
     else:need(canonical and len(prompt)+budget>512,'capacity fallback')
    else:
     need(canonical and trace['roundtrip'] and trace['prompt_ids']==prompt,'noncausal conditioning')
     generated=trace['generated_ids'];need(len(generated)<=budget and not any(t in cfg['draft_stops'] for t in generated),'draft length')
     text=draft.decode(generated,skip_special_tokens=False);need(text==trace['draft_text'],'draft text')
     translated=target.encode(prefix+text,add_special_tokens=False).ids if generated else []
     need(translated==trace['translated_ids'],'translation')
     aligned=translated[:cursor]==ids[:cursor];need(trace['aligned']==aligned,'alignment')
     need(proposal==(translated[cursor:cursor+min(depth,remaining-1)] if aligned else []),'proposal')
     common=0
     while common<min(len(history),len(prompt)) and history[common]==prompt[common]:common+=1
     kept=common-1 if common==len(prompt) and common<len(history) else common;counter+=1
     st=trace['reconcile'];need(st['calls']==counter and st['previous_tokens']==len(history),'counter/history')
     need(st['common_tokens']==common and st['retained_tokens']==kept==trace['cached_tokens'],'LCP')
     zero=(2*28*8*(len(history)-kept)*128+1024)*4 if kept<len(history) else 0
     need(st['zeroed_bytes']==zero and st['replay_last_token']==(kept<common) and st['host_snapshot_bytes']==0,'clear/replay')
     missing=len(prompt)-kept;need(trace['missing_tokens']==missing,'missing input')
     chunks=[];left=missing
     while left>0:
      width=1
      while width<left and width<16:width*=2
      chunks.append(width);left-=min(width,left)
     need(chunks==trace['prefill_chunks'],'chunk plan')
     executed=trace['executed_draft_ids'];need(executed==generated[:len(executed)] and len(executed)==len(trace['decode_samples']),'executed draft')
     need(all(goodtime(s['wall_seconds']) and goodtime(s['gpu_ms']) for s in trace['decode_samples']),'draft samples')
     need(trace['prefix_cache_stats']['entries']==trace['prefix_cache_stats']['accounted_host_bytes']==0,'host cache')
     history=prompt+executed
    matched=0
    for token in proposal:
     if cursor+matched>=len(ids) or token!=ids[cursor+matched]:break
     matched+=1
     if token in cfg['target_stops']:break
    count=min(matched+1,remaining,len(ids)-cursor);emitted=ids[cursor:cursor+count]
    for i,token in enumerate(emitted):
     if token in cfg['target_stops']:emitted=emitted[:i+1];break
    need(step['emitted_ids']==emitted and step['consumed']==len(emitted) and step['accepted_draft']==len(emitted)-1,'oracle acceptance')
    cursor+=len(emitted);previous_accepted=len(emitted)-1
   need(cursor==len(ids) and ids[-1] in cfg['target_stops'],'missing output')
   if policy==3:
    reference=next(r for r in baseline['records'] if r['case_id']==case['id'] and r['mode']=='control')
    need(len(reference['rounds'])==len(row['rounds']),'native4 oracle call count differs')
    for actual,ref in zip(row['rounds'],reference['rounds']):
     for key in ('committed_before','proposal','emitted_ids','consumed','accepted_draft'):need(actual[key]==ref[key],'native4 oracle trace differs')
     for key in ('prompt_ids','generated_ids','executed_draft_ids'):need(actual['draft'].get(key)==ref['draft'].get(key),'native4 draft trajectory differs')
   summary.append(dict(case_id=case['id'],depth=policy,target_calls=len(row['rounds']),accepted=sum(t['accepted_draft'] for t in row['rounds']),proposed=sum(len(t['proposal']) for t in row['rounds']),draft_steps=sum(len(t['draft'].get('executed_draft_ids',[])) for t in row['rounds']),missing_tokens=sum(t['draft'].get('missing_tokens',0) for t in row['rounds']),tokens=len(ids)-1,draft_only_seconds=row['draft_only_seconds']))
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):need(data['cleanup'][k]==0,'leak')
 need(data['cleanup']['metal_allocated_bytes']<=4*1024*1024,'close allocation')
 return summary

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--adaptive',action='store_true');a=p.parse_args();r=a.root
 manifest=json.loads((r/('adaptive-manifest3.json' if a.adaptive else 'oracle-manifest2.json')).read_text())
 for name,h in manifest['files'].items():need(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'frozen file changed')
 baseline_path=r.parent/'mistral-draft-head-profile1/generation-result2.json';baseline=json.loads(baseline_path.read_text())
 result=r/('adaptive-result3.json' if a.adaptive else 'oracle-result2.json')
 data=json.loads(result.read_text());need(data['library']==manifest['argv'][3]+'/libgarnet.dylib','loaded library')
 summary=validate(data,baseline,a.adaptive);rejected=[]
 mutations={'incomplete':lambda d:d.update(completed=False),'cursor':lambda d:d['records'][0]['rounds'][0].update(committed_before=2),'acceptance':lambda d:d['records'][0]['rounds'][0].update(consumed=99),'proposal':lambda d:d['records'][0]['rounds'][0]['proposal'].__setitem__(0,0),'future_condition':lambda d:d['records'][0]['rounds'][0]['draft']['prompt_ids'].__setitem__(-1,0),'wrong_lcp':lambda d:d['records'][0]['rounds'][0]['draft']['reconcile'].update(common_tokens=999),'host_copy':lambda d:d['records'][0]['rounds'][0]['draft']['reconcile'].update(host_snapshot_bytes=1),'shape':lambda d:d['records'][0]['rounds'][0]['draft'].update(prefill_chunks=[7]),'budget':lambda d:d['records'][0]['rounds'][0].update(draft_budget=999) if a.adaptive else d['records'][1].update(draft_budget=8),'wrong_target':lambda d:d['records'][0]['generated_ids'].__setitem__(5,0),'leak':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1)}
 for name,fn in mutations.items():
  changed=copy.deepcopy(data);fn(changed)
  try:validate(changed,baseline,a.adaptive)
  except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
  else:raise ValueError('corruption accepted: '+name)
 with (r/('adaptive-audit3.json' if a.adaptive else 'oracle-audit4.json')).open('x') as f:json.dump(dict(completed=True,scope=__doc__,cases=summary,rejected=rejected,oracle_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),native4_reference_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
 print('oracle audit PASS',len(summary),len(rejected))
if __name__=='__main__':main()
