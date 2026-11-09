#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit paired full native Mistral outputs with two independently pinned drafts.

The common causal validator independently retokenizes every proposed prefix,
checks exact greedy target acceptance and full KV/hidden against frozen native
witnesses. Drafts may propose different tokens. No HTTP or coding-quality claim.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from jinja2.sandbox import SandboxedEnvironment
from audit_mistral_prediction_confidence import validate
def need(value,message):
 if not value:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'fresh audit required');root=a.root.resolve()
 m=json.loads((root/'comparison-manifest1.json').read_text())
 for filename,wanted in m['files'].items():need(sha(filename)==wanted,'changed bound file: '+filename)
 reference=json.loads(Path(m['reference']).read_text());fixture=json.loads(Path(m['fixture']).read_text());results={};summaries=[];raw_hashes={};rejections=[]
 for cohort,role in m['sequence']:
  label=f'cohort{cohort}-{role}';path=root/(label+'-result1.json');data=json.loads(path.read_text());inputs=json.loads(Path(m['inputs'][role]).read_text());family='qwen3' if role=='control' else 'qwen2'
  need(data['inputs']==inputs and data['library']==m['library'],'changed input/library')
  need(data['projection_occupancy']=='occupancy-control','changed target projection control')
  need(fixture['model']==inputs['target_model'] and fixture['stops']==inputs['target_stops'],'changed target')
  tc=json.loads((Path(inputs['draft_model'])/'tokenizer_config.json').read_text());template=SandboxedEnvironment().from_string(tc['chat_template'])
  for case,old in zip(inputs['cases'],fixture['cases'][:8]):
   need(case['id']==old['id'] and case['prompt_ids']==old['prompt_ids'] and case['rendered']==old['rendered'],'changed frozen target prompt')
   need(case['draft_prompt']==template.render(messages=old['messages'],add_generation_prompt=True,enable_thinking=False,tools=None),'changed draft prompt')
  rows=validate(data,reference,draft_family=family);need(len(rows)==8,'wrong coverage')
  tokens=sum(len(x['generated_ids'])-1 for x in data['records']);seconds=sum(x['decode_seconds'] for x in data['records'])
  summaries.append(dict(cohort=cohort,role=role,tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,
   draft_seconds=sum(x['draft_seconds'] for x in rows),target_seconds=sum(x['target_seconds'] for x in rows),
   target_calls=sum(x['target_calls'] for x in rows),accepted=sum(x['accepted'] for x in rows),proposed=sum(x['proposed'] for x in rows),draft_steps=sum(x['draft_steps'] for x in rows),prepared_metal_bytes=data['prepared']['memory']['metal_allocated_bytes']))
  results[(cohort,role)]=data;raw_hashes[str(path)]=sha(path)
  def bad_prefix(d):
   x=next(x for row in d['records'] for x in row['rounds'] if x['draft'].get('prompt_ids'));x['draft']['prompt_ids'].append(1)
  def bad_proposal(d):
   x=next(x for row in d['records'] for x in row['rounds'] if x['proposal']);x['proposal'][0]+=1
  mutations={
   'incomplete':lambda d:d.update(completed=False),
   'wrong_revision':lambda d:d['inputs'].update(draft_revision='bad'),
   'missing_case':lambda d:d['records'].pop(),
   'changed_target':lambda d:d['records'][0]['generated_ids'].__setitem__(0,123),
   'changed_state':lambda d:d['records'][0]['state_sha256'][0].__setitem__(0,'0'*64),
   'changed_hidden':lambda d:d['records'][0].update(hidden_sha256='0'*64),
   'future_prefix':bad_prefix,'bad_proposal':bad_proposal,
   'wrong_cursor':lambda d:d['records'][0]['rounds'][0].update(committed_before=2),
   'nonfinite_time':lambda d:d['records'][0].update(decode_seconds=float('nan')),
   'wrong_rate':lambda d:d['records'][0].update(decode_tokens_per_second=1.),
   'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1),
   'memory_budget':lambda d:d['prepared']['memory'].update(metal_allocated_bytes=2**40),
  }
  for name,mutate in mutations.items():
   bad=copy.deepcopy(data);mutate(bad)
   try:validate(bad,reference,draft_family=family)
   except ValueError:rejections.append(label+':'+name)
   else:raise ValueError('accepted corruption: '+name)
 comparisons=[]
 for cohort in (1,2):
  control,candidate=results[(cohort,'control')],results[(cohort,'candidate')];ratios=[]
  for left,right in zip(control['records'],candidate['records']):
   for key in ('case_id','prompt_ids','generated_ids','slots','state_sha256','hidden_sha256'):need(left[key]==right[key],'target differs between drafts: '+key)
   ratios.append(dict(case_id=left['case_id'],ratio=left['decode_seconds']/right['decode_seconds']))
  old=next(x for x in summaries if x['cohort']==cohort and x['role']=='control');new=next(x for x in summaries if x['cohort']==cohort and x['role']=='candidate')
  comparisons.append(dict(cohort=cohort,speed_ratio=old['seconds']/new['seconds'],per_case=ratios))
 out=dict(completed=True,scope=__doc__,summaries=summaries,comparisons=comparisons,rejected_corruptions=rejections,raw_files=raw_hashes,all_target_ids_kv_hidden_match_frozen_native=True,strict_coding_score_unchanged='0/8',matched_http_speedup_admitted=False)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('completed','summaries','comparisons')},indent=2))
if __name__=='__main__':main()
