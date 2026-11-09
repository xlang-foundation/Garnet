#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit the separate upstream v3 prompt experiment and unchanged quality gate.

Legacy baselines remain valid only for their original prompts. This checks
protocol/token decoding, complete collection and evaluation provenance; there
is no new reference-engine generation parity or comparative performance claim.
"""
import argparse,copy,hashlib,json,math
from pathlib import Path
import sentencepiece as sp
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

def need(value,why):
 if not value:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(s):return isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s)
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
def validate(status,rows,fixture,tokenizer,sentencepiece,library):
 need(status['completed'] is True and status['library']==library,'incomplete/wrong library')
 need(status['backend'].startswith('Garnet Metal/MPS '),'wrong backend')
 need(len(status['records'])==len(rows)==len(fixture['cases'])==8,'missing/extra run')
 need(status['prepared']['graphs']==224,'changed prepared family')
 for summary,row,case in zip(status['records'],rows,fixture['cases']):
  ids=row['generated_ids']
  need(summary['case_id']==case['id'],'changed case order')
  need(row['status']=='completed' and row['protocol']==fixture['protocol'],'wrong result/protocol')
  need(row['prompt_ids']==case['prompt_ids'],'changed prompt IDs')
  need(all(type(t) is int and 0<=t<32768 for t in ids) and 1<=len(ids)<=1024,'bad output ID/budget')
  need(ids[-1]==2 and 2 not in ids[:-1] and row['finish_reason']=='stop','incorrect stopping')
  need(summary['output_tokens']==len(ids) and summary['finish_reason']==row['finish_reason'],'wrong count/status')
  need(row['text']==tokenizer.decode(ids,skip_special_tokens=True)==sentencepiece.decode(ids),'changed decoded text')
  need(row['slots']==[len(case['prompt_ids'])+len(ids)-1],'incorrect KV cursor')
  need(finite(row['prefill_seconds']) and finite(row['decode_seconds']),'invalid time')
  need(len(row['state_sha256'])==32 and all(len(s)==2 and all(digest(h) for h in s) for s in row['state_sha256']) and digest(row['hidden_sha256']),'incomplete state hashes')
 for memory in [status['prepared']['memory']]+[row['memory'] for row in rows]:
  need(0<=memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-512*1024*1024,'prepared/run memory budget')
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  need(status['cleanup'][key]==0,'live owner')
 need(0<=status['cleanup']['metal_allocated_bytes']<=4*1024*1024,'close allocation')
 return True
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--quality',type=Path,required=True);p.add_argument('--evaluator',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();need(not a.output.exists(),'fresh output required')
 root=a.root.resolve();manifest=json.loads((root/'native-manifest1.json').read_text())
 for filename,wanted in manifest['files'].items():need(sha(filename)==wanted,'changed bound file: '+filename)
 fixture=json.loads((root/'fixture1/fixture.json').read_text());suite=json.loads((root/'fixture1/suite.json').read_text())
 need(fixture['protocol']=='mistral_v3_upstream_single_user_text_v1' and fixture['suite_sha256']==sha(root/'fixture1/suite.json'),'wrong suite/profile')
 tc=json.loads((root/'upstream/tokenizer_config.json').read_text());env=ImmutableSandboxedEnvironment();env.globals['raise_exception']=lambda message:(_ for _ in ()).throw(ValueError(message));template=env.from_string(tc['chat_template'])
 tokenizer=Tokenizer.from_file(str(root/'upstream/tokenizer.json'));sentencepiece=sp.SentencePieceProcessor(model_file=str(root/'upstream/tokenizer.model.v3'))
 for case,task in zip(fixture['cases'],suite['tasks']):
  need(case['id']==task['id'] and case['messages']==[dict(role='user',content=task['prompt'])] and case['max_tokens']==1024,'changed request')
  text=template.render(messages=case['messages'],bos_token='<s>',eos_token='</s>',add_generation_prompt=True,tools=None)
  need(case['rendered']==text,'changed template rendering')
  need(case['prompt_ids']==tokenizer.encode(text,add_special_tokens=False).ids==[1,3]+sentencepiece.encode(task['prompt'],out_type=int)+[4],'wrong upstream prompt IDs')
  need(case['legacy_prompt_ids']!=case['prompt_ids'] and len(case['legacy_prompt_ids'])==len(case['prompt_ids'])+2,'legacy framing unexpectedly identical')
 directory=root/'native1';status=json.loads((directory/'status.json').read_text());paths=[directory/c['id']/'request-0.result.json' for c in fixture['cases']];rows=[json.loads(p.read_text()) for p in paths];library=manifest['argv'][3]+'/libgarnet.dylib'
 validate(status,rows,fixture,tokenizer,sentencepiece,library)
 quality=json.loads(a.quality.read_text());need(quality['sandbox_probe']['status']=='passed','evaluator infrastructure failure')
 need(quality['suite_sha256']==fixture['suite_sha256'] and quality['evaluator_sha256']==sha(a.evaluator),'changed evaluator/contracts')
 need(len(quality['results'])==quality['total']==8,'missing quality result')
 for result,case,path in zip(quality['results'],fixture['cases'],paths):need(result['task']==case['id'] and result['response_sha256']==sha(path),'quality/output identity mismatch')
 need(quality['passed']==sum(r['status']=='passed' for r in quality['results']),'wrong quality arithmetic')
 rejected=[]
 mutations={
  'incomplete':lambda s,r:s.update(completed=False),
  'wrong_library':lambda s,r:s.update(library='other'),
  'wrong_order':lambda s,r:s['records'][0].update(case_id='other'),
  'missing_run':lambda s,r:r.pop(),
  'legacy_prompt':lambda s,r:r[0].update(prompt_ids=fixture['cases'][0]['legacy_prompt_ids']),
  'wrong_token':lambda s,r:r[0]['generated_ids'].__setitem__(0,32768),
  'bool_token':lambda s,r:r[0]['generated_ids'].__setitem__(0,True),
  'early_eos':lambda s,r:r[0]['generated_ids'].__setitem__(0,2),
  'changed_text':lambda s,r:r[0].update(text='changed'),
  'wrong_slot':lambda s,r:r[0]['slots'].__setitem__(0,1),
  'nonfinite_time':lambda s,r:r[0].update(decode_seconds=float('nan')),
  'missing_state':lambda s,r:r[0]['state_sha256'].pop(),
  'bad_hidden':lambda s,r:r[0].update(hidden_sha256='bad'),
  'wrong_count':lambda s,r:s['records'][0].update(output_tokens=1),
  'live_owner':lambda s,r:s['cleanup'].update(garnet_live_matrix_owners=1),
  'over_budget':lambda s,r:s['prepared']['memory'].update(metal_allocated_bytes=2**40),
 }
 for name,mutate in mutations.items():
  s,r=copy.deepcopy(status),copy.deepcopy(rows);mutate(s,r)
  try:validate(s,r,fixture,tokenizer,sentencepiece,library)
  except ValueError:rejected.append(name)
  else:raise ValueError('accepted corrupt evidence: '+name)
 out=dict(completed=True,scope=__doc__,protocol=fixture['protocol'],upstream_revision=fixture['upstream_revision'],native_decoding_matches_both_independent_decoders=True,
  quality_passed=quality['passed'],quality_total=quality['total'],output_tokens=sum(len(r['generated_ids']) for r in rows),rejected_corruptions=rejected,
  raw_files={str(p):sha(p) for p in paths+[directory/'status.json',a.quality,a.evaluator]},legacy_baselines_unchanged=True,reference_generation_parity=False,matched_speedup_qualification=False)
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('completed','quality_passed','quality_total','output_tokens')},indent=2))
if __name__=='__main__':main()
