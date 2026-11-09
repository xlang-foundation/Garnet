#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit serial native three-row projection generation against frozen witnesses.

The bound confidence-stage validator independently retokenizes causal draft
traces and checks complete target IDs, full KV/hidden hashes and native routes.
This experiment does not establish HTTP speed, coding quality or a 2x win.
"""
import argparse,copy,hashlib,importlib.util,json,math
from pathlib import Path
from jinja2.sandbox import SandboxedEnvironment

def need(value,why):
 if not value:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('root',type=Path);p.add_argument('--validator',type=Path,required=True)
 p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();need(not args.output.exists(),'refuse to replace audit')
 root=args.root.resolve();manifest=json.loads(args.manifest.read_text())
 for filename,wanted in manifest['files'].items():need(sha(filename)==wanted,'changed bound file: '+filename)
 need(str(args.validator.resolve()) in manifest['files'],'unbound validator')
 spec=importlib.util.spec_from_file_location('frozen_validator',args.validator);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
 fixture_path=root.parent/'mistral-coding-context-fixture1/fixture.json'
 reference_path=root.parent/'mistral-draft-head-profile1/generation-result2.json'
 for path in (fixture_path,reference_path):need(str(path) in manifest['files'],'unbound frozen witness')
 fixture=json.loads(fixture_path.read_text());reference=json.loads(reference_path.read_text())
 inputs=json.loads((root/'generation-input1.json').read_text());need(str(root/'generation-input1.json') in manifest['files'],'unbound input')
 need(fixture['model']==inputs['target_model'] and fixture['stops']==inputs['target_stops'],'changed target')
 tc=json.loads((Path(inputs['draft_model'])/'tokenizer_config.json').read_text())
 template=SandboxedEnvironment().from_string(tc['chat_template'])
 for original,case in zip(fixture['cases'][:8],inputs['cases']):
  need(case['id']==original['id'] and case['rendered']==original['rendered'] and case['prompt_ids']==original['prompt_ids'],'changed fixture')
  need(case['draft_prompt']==template.render(messages=original['messages'],add_generation_prompt=True,enable_thinking=False,tools=None),'changed draft prompt')
 results={};summaries=[];rejected=[];bound_results={}
 for cohort,engine in manifest['sequence']:
  label=f'cohort{cohort}-{engine}';path=root/(label+'-result2.json');data=json.loads(path.read_text())
  need(data['inputs']==inputs,'changed run input')
  need(data['library']==manifest['libraries'][engine],'wrong measured library')
  need(data['projection_occupancy']=='occupancy-'+engine,'wrong engine role')
  report=v.validate(data,reference);need(len(report)==8,'wrong coverage')
  for row in data['records']:
   for step in row['rounds']:
    if step['proposal']:
     need(step['metrics']['target_indexed_decode_dispatches']==(224 if len(step['proposal'])<3 else 0),'wrong physical/selected row route')
  results[(cohort,engine)]=data;bound_results[str(path)]=sha(path)
  tokens=sum(len(row['generated_ids'])-1 for row in data['records']);seconds=sum(row['decode_seconds'] for row in data['records'])
  occupancy={str(n):sum(len(x['proposal'])+1==n for row in data['records'] for x in row['rounds'] if x['proposal']) for n in (2,3,4)}
  summaries.append(dict(cohort=cohort,engine=engine,tokens=tokens,decode_seconds=seconds,tokens_per_second=tokens/seconds,
   target_seconds=sum(x['target_seconds'] for row in data['records'] for x in row['rounds']),
   draft_seconds=sum(x['draft']['seconds'] for row in data['records'] for x in row['rounds']),verification_occupancies=occupancy))
  def route(d):
   step=next(x for row in d['records'] for x in row['rounds'] if len(x['proposal'])==2)
   step['metrics']['target_affine_decode_block_dispatches']+=1
  def future(d):
   step=next(x for row in d['records'] for x in row['rounds'] if x['draft'].get('prompt_ids'))
   step['draft']['prompt_ids'].append(1)
  def proposal(d):
   step=next(x for row in d['records'] for x in row['rounds'] if x['proposal'])
   step['proposal'][0]+=1
  mutations={
   'incomplete':lambda d:d.update(completed=False),
   'wrong_engine':lambda d:d.update(projection_occupancy='occupancy-'+('control' if engine=='candidate' else 'candidate')),
   'changed_input':lambda d:d['inputs'].update(capacity=2048),
   'missing_run':lambda d:d['records'].pop(),
   'wrong_route':route,
   'target_id':lambda d:d['records'][0]['generated_ids'].__setitem__(0,123),
   'state':lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64),
   'hidden':lambda d:d['records'][0].update(hidden_sha256='0'*64),
   'future_prefix':future,
   'proposal':proposal,
   'cursor':lambda d:d['records'][0]['rounds'][0].update(committed_before=2),
   'nonfinite_time':lambda d:d['records'][0].update(decode_seconds=float('nan')),
   'bad_token_rate':lambda d:d['records'][0].update(decode_tokens_per_second=1.),
   'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
   'budget':lambda d:d['prepared']['memory'].update(metal_allocated_bytes=2**40),
  }
  for name,mutate in mutations.items():
   bad=copy.deepcopy(data);mutate(bad)
   try:v.validate(bad,reference)
   except ValueError:rejected.append(label+':'+name)
   else:raise ValueError('accepted corruption: '+label+':'+name)
 comparisons=[]
 for cohort in (1,2):
  control=results[(cohort,'control')];candidate=results[(cohort,'candidate')];ratios=[]
  for left,right in zip(control['records'],candidate['records']):
   for key in ('case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256'):
    need(left[key]==right[key],'paired target changed: '+key)
   need(len(left['rounds'])==len(right['rounds']),'paired target calls differ')
   for a,b in zip(left['rounds'],right['rounds']):
    for key in ('committed_before','slot','proposal','submitted_ids','emitted_ids','consumed','accepted_draft'):
     need(a[key]==b[key],'paired proposal/commit differs: '+key)
    for key in ('prompt_ids','generated_ids','executed_draft_ids','cached_tokens','prefill_chunks','translated_ids'):
     need(a['draft'].get(key)==b['draft'].get(key),'paired draft differs: '+key)
   ratios.append(dict(case_id=left['case_id'],ratio=left['decode_seconds']/right['decode_seconds']))
  a=next(x for x in summaries if x['cohort']==cohort and x['engine']=='control')
  b=next(x for x in summaries if x['cohort']==cohort and x['engine']=='candidate')
  comparisons.append(dict(cohort=cohort,ratio=a['decode_seconds']/b['decode_seconds'],per_case=ratios))
 out=dict(completed=True,scope=__doc__,summaries=summaries,comparisons=comparisons,rejected_corruptions=rejected,
  frozen_files=manifest['files'],raw_results=bound_results,validator=str(args.validator.resolve()),
  all_target_ids_full_kv_hidden_and_draft_proposals_identical=True,
  coding_quality_admitted=False,matched_http_speedup_admitted=False)
 args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('completed','summaries','comparisons')},indent=2))
if __name__=='__main__':main()
