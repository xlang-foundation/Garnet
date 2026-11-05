#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mutate real assets to verify unsupported formats fail during native open."""
import argparse,copy,hashlib,json,pathlib,subprocess

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=pathlib.Path,required=True)
 p.add_argument('--binary',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True)
 a=p.parse_args();a.output.mkdir(exist_ok=False)
 original=json.loads((a.model/'tokenizer.json').read_text())
 cases=[('unigram',lambda t:t['model'].update(type='Unigram')),
        ('dropout',lambda t:t['model'].update(dropout=0.1)),
        ('unknown-normalizer',lambda t:t.update(normalizer={'type':'NFKC'})),
        ('unknown-decoder',lambda t:t.update(decoder={'type':'UnknownDecoder'})),
        ('single-word-added-token',lambda t:t['added_tokens'][0].update(single_word=True)),
        ('strip-added-token',lambda t:t['added_tokens'][0].update(lstrip=True)),
        ('truncation',lambda t:t.update(truncation={'max_length':4})),
        ('prefix-space',lambda t:t['pre_tokenizer']['pretokenizers'][-1].update(add_prefix_space=True)),
        ('invert-regex',lambda t:t['pre_tokenizer']['pretokenizers'][0].update(invert=True)),
        ('unknown-postprocessor',lambda t:t.update(post_processor={'type':'UnknownProcessor'}))]
 rows=[]
 for name,mutate in cases:
  folder=a.output/name;folder.mkdir();root=copy.deepcopy(original);mutate(root)
  (folder/'tokenizer.json').write_text(json.dumps(root,ensure_ascii=True))
  for filename in ('config.json','tokenizer_config.json'):(folder/filename).symlink_to((a.model/filename).resolve())
  r=subprocess.run([str(a.binary),str(folder)],capture_output=True,text=True,timeout=30)
  rows.append(dict(case=name,returncode=r.returncode,error=r.stderr.strip(),rejected=r.returncode==1 and bool(r.stderr)))
  (folder/'tokenizer.json').unlink()
 report=dict(passed=all(x['rejected'] for x in rows),cases=rows,
             binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
             source_tokenizer_sha256=hashlib.sha256((a.model/'tokenizer.json').read_bytes()).hexdigest())
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
 raise SystemExit(0 if report['passed'] else 1)
if __name__=='__main__':main()
