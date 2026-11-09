#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit B4 early-finish correctness and isolated three-survivor command timing.

Timing excludes prefill, scheduler, tensor preparation and KV hashing. Do not
compare this rate directly with HTTP aggregate throughput or decode rates.
"""
import argparse,copy,hashlib,json,math
from pathlib import Path
def need(value,why):
 if not value:raise ValueError(why)
def digest(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def positive(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>0
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate(data,inputs,engine):
 need(data['completed'] is True and data['engine']==engine,'incomplete/wrong engine')
 need(('mistral_decode_reuse_three_rows' in data['backend'])==(engine=='candidate'),'wrong compiled engine')
 need(len(data['records'])==4,'missing request')
 for index,(row,case) in enumerate(zip(data['records'],inputs['cases'][:4])):
  control=json.loads(Path(case['control']).read_text());budget=2 if index==0 else 128
  need(row['case_id']==case['id'] and row['row']==index and row['epoch']>0,'wrong owner/case')
  need(row['generated_ids']==row['expected_ids']==control['generated_ids'][:budget],'target IDs differ')
  need(len(row['state_sha256'])==32 and all(len(layer)==2 and all(digest(h) for h in layer) for layer in row['state_sha256']),'invalid full KV hash')
  need(digest(row['hidden_sha256']),'invalid hidden hash')
 need(data['inactive_row_checks']==2 and data['inactive_before']==data['inactive_after'],'inactive row changed')
 need(len(data['inactive_before'])==32 and all(len(x)==2 and all(digest(h) for h in x) for x in data['inactive_before']),'missing inactive KV')
 samples=data['samples'];need(len(samples)==127,'wrong command count')
 epochs=[r['epoch'] for r in data['records']]
 for index,s in enumerate(samples):
  active=list(range(4)) if index==0 else [1,2,3]
  need(s['active_rows']==active,'wrong active rows')
  need(s['epochs']==[epochs[i] if i in active else 0 for i in range(4)],'wrong lease epoch')
  need(positive(s['seconds']) and positive(s['gpu_ms']),'invalid timing')
  need(s['affine_blocks']==(161 if index==0 or engine=='candidate' else 0),'wrong blocked route')
 need(data['prepared']['metal_allocated_bytes']<=data['prepared']['metal_recommended_working_set_bytes']-512*1024*1024,'prepared memory budget')
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  need(data['cleanup'][key]==0,'live owner')
 need(0<=data['cleanup']['metal_allocated_bytes']<=4*1024*1024,'close allocation')
 steady=samples[1:];seconds=sum(s['seconds'] for s in steady);tokens=3*len(steady)
 return dict(steady_commands=len(steady),steady_tokens=tokens,command_seconds=seconds,command_tokens_per_second=tokens/seconds,gpu_ms=sum(s['gpu_ms'] for s in steady),complete_generated_ids=sum(len(r['generated_ids']) for r in data['records']))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();need(not args.output.exists(),'refuse overwrite')
 root=args.root.resolve();manifest=json.loads(args.manifest.read_text());attempt=manifest['attempt'];need(type(attempt) is int and attempt>0,'invalid attempt')
 for filename,wanted in manifest['files'].items():need(sha(filename)==wanted,'changed bound file: '+filename)
 inputs=json.loads((root/'generation-input1.json').read_text());summaries=[];results={};hashes={};rejected=[]
 for cohort,engine in manifest['sequence']:
  label=f'batch-cohort{cohort}-{engine}';path=root/(label+f'-result{attempt}.json');data=json.loads(path.read_text());need(data['library']==manifest['libraries'][engine],'wrong library')
  summary=validate(data,inputs,engine);summary.update(cohort=cohort,engine=engine);summaries.append(summary);results[(cohort,engine)]=data;hashes[str(path)]=sha(path)
  mutations={
   'incomplete':lambda d:d.update(completed=False),
   'missing_request':lambda d:d['records'].pop(),
   'wrong_case':lambda d:d['records'][0].update(case_id='other'),
   'wrong_target':lambda d:d['records'][1]['generated_ids'].__setitem__(0,7),
   'missing_state':lambda d:d['records'][1]['state_sha256'].pop(),
   'missing_hidden':lambda d:d['records'][1].update(hidden_sha256='bad'),
   'inactive_changed':lambda d:d['inactive_after'][0].__setitem__(0,'0'*64),
   'inactive_not_checked':lambda d:d.update(inactive_row_checks=0),
   'wrong_sample_count':lambda d:d['samples'].pop(),
   'wrong_active_row':lambda d:d['samples'][1].update(active_rows=[0,1,2]),
   'stale_epoch':lambda d:d['samples'][1]['epochs'].__setitem__(1,999),
   'nonfinite_time':lambda d:d['samples'][1].update(seconds=float('nan')),
   'wrong_route':lambda d:d['samples'][1].update(affine_blocks=99),
   'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
  }
  for name,mutate in mutations.items():
   bad=copy.deepcopy(data);mutate(bad)
   try:validate(bad,inputs,engine)
   except ValueError:rejected.append(label+':'+name)
   else:raise ValueError('accepted bad evidence: '+name)
 comparisons=[]
 for cohort in (1,2):
  left,right=results[(cohort,'control')],results[(cohort,'candidate')]
  for a,b in zip(left['records'],right['records']):
   for key in ('case_id','row','generated_ids','expected_ids','state_sha256','hidden_sha256'):need(a[key]==b[key],'paired correctness differs: '+key)
  a=next(x for x in summaries if x['cohort']==cohort and x['engine']=='control');b=next(x for x in summaries if x['cohort']==cohort and x['engine']=='candidate')
  comparisons.append(dict(cohort=cohort,speed_ratio=a['command_seconds']/b['command_seconds']))
 out=dict(completed=True,scope=__doc__,summaries=summaries,comparisons=comparisons,rejected_corruptions=rejected,raw_results=hashes,all_ids_kv_hidden_identical=True,matched_serving_performance=False)
 args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('completed','summaries','comparisons')},indent=2))
if __name__=='__main__':main()
