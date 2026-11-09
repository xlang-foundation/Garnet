#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze a reproducible, text-only native Mistral / Qwen draft experiment.

No inference runs here. The CPU template renderer uses pinned checkpoint data;
native Garnet C++ tokenizers and Metal kernels execute in the separate runner.
"""
import argparse,hashlib,json
from pathlib import Path
from jinja2.sandbox import SandboxedEnvironment

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('fixture','controls','draft-model','library','source','runner','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();root=a.output.resolve();root.mkdir()
 fixture=json.loads(a.fixture.read_text());draft=a.draft_model.resolve();manifest=json.loads((draft/'download-manifest.json').read_text())
 if manifest['repo']!='Qwen/Qwen3-0.6B' or manifest['revision']!='c1899de289a04d12100db370d81485cdf75e47ca':raise ValueError('Unrecognized draft artifact')
 for entry in manifest['files']:
  path=draft/entry['file']
  if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('Changed draft asset: '+entry['file'])
 config=json.loads((draft/'tokenizer_config.json').read_text());template=SandboxedEnvironment().from_string(config['chat_template']);cases=[]
 for row in fixture['cases'][:8]:
  control=a.controls.resolve()/row['id']/'request-0.result.json'
  if row['kind']!='coding' or len(row['prompt_ids'])+64>512 or not control.is_file():raise ValueError('Unadmitted coding fixture')
  prompt=template.render(messages=row['messages'],add_generation_prompt=True,enable_thinking=False,tools=None)
  if not prompt.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'):raise ValueError('Changed Qwen chat template')
  cases.append(dict(id=row['id'],rendered=row['rendered'],prompt_ids=row['prompt_ids'],control=str(control),draft_prompt=prompt))
 if len(cases)!=8:raise ValueError('Expected eight coding cases')
 inputs=dict(target_model=fixture['model'],draft_model=str(draft),draft_revision=manifest['revision'],draft_tokenizer_sha256=next(x['sha256'] for x in manifest['files'] if x['file']=='tokenizer.json'),target_stops=fixture['stops'],draft_stops=[151643,151645],capacity=512,output_limit=64,draft_token_limit=8,cases=cases)
 ip=root/'input.json';ip.write_text(json.dumps(inputs,indent=2)+'\n')
 files=[ip,a.fixture.resolve(),a.runner.resolve(),Path(__file__).resolve(),draft/'download-manifest.json']+[draft/x['file'] for x in manifest['files']]+[Path(c['control']) for c in cases]
 library=a.library.resolve()/'libgarnet.dylib'
 raw=dict(files={str(path):sha(path) for path in files},library=str(library),library_sha256=sha(library),source=str(a.source.resolve()),policy='Three serial modes (ordinary/fixed16/compact) reverse order by case; one clipped B1/C512 sequence per mode/case, no warmup, statistical or HTTP claim.')
 (root/'manifest.json').write_text(json.dumps(raw,indent=2)+'\n')
 (root/'command.json').write_text(json.dumps(dict(argv=['xlang3','-B',str(a.runner.resolve()),str(library.parent),str(a.source.resolve()),str(ip),str(root/'result.json')]),indent=2)+'\n')
 print('prepared',len(cases),'coding cases',root)
if __name__=='__main__':main()
