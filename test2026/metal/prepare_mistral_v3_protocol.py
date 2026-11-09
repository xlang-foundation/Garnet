#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze an explicit upstream Mistral v3 single-user text protocol.

Preserves the converted asset and all legacy fixtures. The current upstream
template and independent SentencePiece v3 instruction IDs must agree. This
does not qualify tools, multimodal messages or general conversation rendering.
"""
import argparse,hashlib,json
from pathlib import Path
import sentencepiece as sp
from tokenizers import Tokenizer
from jinja2.sandbox import ImmutableSandboxedEnvironment

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(value,why):
 if not value:raise ValueError(why)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('model','upstream','upstream-manifest','legacy-fixture','suite','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();need(not a.output.exists(),'fresh output required')
 manifest=json.loads(a.upstream_manifest.read_text())
 need(manifest['repo']=='mistralai/Mistral-7B-Instruct-v0.3','wrong upstream')
 for name,entry in manifest['files'].items():need(sha(a.upstream/name)==entry['sha256'],'changed upstream asset')
 need(sha(a.model/'tokenizer.json')==sha(a.upstream/'tokenizer.json'),'different tokenizer pipeline')
 need(sha(a.model/'tokenizer.model')==sha(a.upstream/'tokenizer.model.v3'),'different SentencePiece model')
 legacy=json.loads(a.legacy_fixture.read_text());suite=json.loads(a.suite.read_text())
 cases=[c for c in legacy['cases'] if c['kind']=='coding'];need(len(cases)==len(suite['tasks'])==8,'wrong suite coverage')
 need(Path(legacy['model']).resolve()==a.model.resolve() and legacy['stops']==[2],'wrong model/stops')
 config=json.loads((a.upstream/'tokenizer_config.json').read_text())
 env=ImmutableSandboxedEnvironment();env.globals['raise_exception']=lambda message:(_ for _ in ()).throw(ValueError(message))
 template=env.from_string(config['chat_template']);tokenizer=Tokenizer.from_file(str(a.upstream/'tokenizer.json'))
 native_reference=sp.SentencePieceProcessor(model_file=str(a.upstream/'tokenizer.model.v3'))
 need(native_reference.bos_id()==1 and native_reference.eos_id()==2,'wrong v3 control IDs')
 output=[]
 for case,task in zip(cases,suite['tasks']):
  need(case['id']==task['id'] and case['messages']==[dict(role='user',content=task['prompt'])],'changed coding request')
  rendered=template.render(messages=case['messages'],bos_token='<s>',eos_token='</s>',add_generation_prompt=True,tools=None)
  ids=tokenizer.encode(rendered,add_special_tokens=False).ids
  direct=[1,3]+native_reference.encode(task['prompt'],out_type=int)+[4]
  need(ids==direct,'upstream template/SentencePiece instruction mismatch')
  need(case['max_tokens']==suite['sampling']['max_tokens']==1024 and len(ids)+1024<=4096,'changed budget/profile')
  output.append(dict(case,id=case['id'],legacy_rendered=case['rendered'],legacy_prompt_ids=case['prompt_ids'],rendered=rendered,prompt_ids=ids,prompt_tokens=len(ids)))
 result=dict(schema=1,model=str(a.model.resolve()),stops=[2],capacity=4096,maximum_prefill_chunk=128,
  protocol='mistral_v3_upstream_single_user_text_v1',upstream_revision=manifest['revision'],
  upstream_template_sha256=hashlib.sha256(config['chat_template'].encode()).hexdigest(),
  legacy_fixture_sha256=sha(a.legacy_fixture),suite_sha256=sha(a.suite),cases=output,scope=__doc__)
 a.output.mkdir();(a.output/'fixture.json').write_text(json.dumps(result,indent=2)+'\n');(a.output/'suite.json').write_bytes(a.suite.read_bytes())
 files=[a.upstream_manifest,a.legacy_fixture,a.suite,a.model/'tokenizer.json',a.model/'tokenizer_config.json',a.model/'tokenizer.model']+list(a.upstream.iterdir())
 (a.output/'manifest.json').write_text(json.dumps(dict(files={str(x.resolve()):sha(x) for x in files if x.is_file()},profile=result['protocol'],source_sha256=sha(__file__)),indent=2)+'\n')
 print(json.dumps(dict(profile=result['protocol'],prompt_tokens=[c['prompt_tokens'] for c in output],unchanged_legacy_assets=True)))
if __name__=='__main__':main()
