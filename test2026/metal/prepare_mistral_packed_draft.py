#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extend frozen text inputs with a verified packed draft and explicit bounds.

Use the original cross-tokenizer preparer first. No inference runs here.
The optional draft bound changes only the auxiliary engine; Mistral staysC4096
for complete-output studies and falls back to ordinary decode when needed.
"""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('base-input','packed-model','library','runner','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--study',choices=('clipped','full'),required=True);p.add_argument('--draft-capacity',type=int,choices=(512,4096));a=p.parse_args()
 raw=json.loads(a.base_input.read_text());folder=a.packed_model.resolve();converted=json.loads((folder/'conversion-manifest.json').read_text())
 if raw['draft_revision']!=converted['source_revision'] or converted['source_repo']!='Qwen/Qwen3-0.6B':raise ValueError('Changed source identity')
 raw_folder=Path(raw['draft_model']);files=[a.base_input.resolve(),folder/'conversion-manifest.json',a.runner.resolve(),Path(__file__).resolve()]
 for row in converted['source_files']:
  path=raw_folder/row['file']
  if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Changed BF16 source asset')
  files.append(path)
 for row in converted['files']:
  path=folder/row['file']
  if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Changed packed asset')
  files.append(path)
 raw['draft_model']=str(folder);raw['draft_checkpoint_format']='mlx_qwen3_affine_int4_g64_fp16'
 if a.study=='clipped':
  if a.draft_capacity is not None:raise ValueError('Clipped comparison has fixedC512 bounds')
  raw['draft_bf16_model']=str(raw_folder)
 else:
  raw['capacity']=4096;raw['output_limit']=1024
  if a.draft_capacity is not None:raw['draft_capacity']=a.draft_capacity
 root=a.output.resolve();root.mkdir();ip=root/'input.json';ip.write_text(json.dumps(raw,indent=2)+'\n');files.append(ip)
 lib=a.library.resolve()/'libgarnet.dylib'
 (root/'manifest.json').write_text(json.dumps(dict(files={str(p):sha(p) for p in files},library=str(lib),library_sha256=sha(lib),study=a.study,draft_capacity=raw.get('draft_capacity',raw['capacity'])),indent=2)+'\n')
 print('prepared',root)
if __name__=='__main__':main()
