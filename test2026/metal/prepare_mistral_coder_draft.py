#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reproduce an explicit pinned Qwen2.5-Coder draft input for native Mistral.

Target requests/budgets remain those of the supplied frozen original fixture.
This only prepares inputs; it does not download assets or execute a model.
"""
import argparse,hashlib,json
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(value,why):
 if not value:raise ValueError(why)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('base-input','model','asset-manifest','fixture','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();need(not a.output.exists(),'fresh output required');manifest=json.loads(a.asset_manifest.read_text())
 need(manifest['complete'] is True and manifest['repo']=='mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit' and manifest['revision']=='6b16732e5af5cd9bd600186ad59fa618867ef7a4','unqualified draft identity')
 need(Path(manifest['model']).resolve()==a.model.resolve(),'model/manifest path mismatch')
 for name,wanted in manifest['files'].items():
  need(Path(name).name==name and name not in ('.','..'),'invalid asset name')
  path=a.model/name;need(path.stat().st_size==wanted['bytes'] and sha(path)==wanted['sha256'],'changed draft asset')
 config=json.loads((a.model/'config.json').read_text());need(config['model_type']=='qwen2' and config['quantization']['bits']==4 and config['quantization']['group_size']==64,'unsupported draft representation')
 inputs=json.loads(a.base_input.read_text());fixture=json.loads(a.fixture.read_text());need(inputs['capacity']==4096 and inputs['output_limit']==1024 and inputs['draft_token_limit']==8,'changed target budgets')
 need(inputs['target_model']==fixture['model'] and inputs['target_stops']==fixture['stops'] and len(inputs['cases'])==8,'changed target identity/coverage')
 tc=json.loads((a.model/'tokenizer_config.json').read_text());env=ImmutableSandboxedEnvironment();env.globals['raise_exception']=lambda message:(_ for _ in ()).throw(ValueError(message));template=env.from_string(tc['chat_template'])
 inputs.update(draft_model=str(a.model.resolve()),draft_revision=manifest['revision'],draft_source_repo=manifest['repo'],draft_tokenizer_sha256=sha(a.model/'tokenizer.json'),draft_checkpoint_format='mlx_qwen2_affine_int4_g64_fp16')
 for old,case in zip(fixture['cases'][:8],inputs['cases']):
  need(case['id']==old['id'] and case['prompt_ids']==old['prompt_ids'] and case['rendered']==old['rendered'],'changed target prompt')
  case['draft_prompt']=template.render(messages=old['messages'],add_generation_prompt=True,enable_thinking=False,tools=None)
 a.output.mkdir();(a.output/'input.json').write_text(json.dumps(inputs,indent=2)+'\n')
 files=[a.base_input,a.asset_manifest,a.fixture,a.model/'config.json',a.model/'tokenizer_config.json',a.model/'tokenizer.json',Path(__file__)]
 (a.output/'manifest.json').write_text(json.dumps(dict(files={str(x.resolve()):sha(x) for x in files},scope=__doc__),indent=2)+'\n')
 print('prepared',a.output)
if __name__=='__main__':main()
