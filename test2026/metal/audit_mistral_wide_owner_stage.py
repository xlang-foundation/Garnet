#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit paired full-prefill measurements; no serving or 2x admission."""
import argparse,hashlib,json,statistics
from pathlib import Path

def require(value,message):
 if not value:raise ValueError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files={}
 def read(path):
  data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
 rows=[];identities={};library=None;measurements=0
 for cohort,reverse in [(1,0),(2,1)]:
  status=read(a.study/f'cohort-{cohort}-status.json');result=read(a.study/f'cohort-{cohort}.json')
  require(status['status']=='terminal' and status['returncode']==0 and status['pid']>0,'unfinished cohort')
  require(result['completed'] is True and result['reverse']==reverse,'incomplete paired study')
  require(len(result['groups'])==2,'missing owner groups')
  if library is None:library=result['library']
  require(result['library']==library,'mixed native libraries')
  for group,live,wide in zip(result['groups'],[[0],[1,2]],[512,256]):
   require(group['live']==live and group['wide_chunk']==wide,'changed owner profiles')
   require([v['chunk'] for v in group['variants']]==([128,wide] if not reverse else [wide,128]),'changed paired order')
   variants={v['chunk']:v for v in group['variants']};medians={}
   for chunk,variant in variants.items():
    require(len(variant['measured'])==3 and [r['repeat'] for r in variant['measured']]==list(range(3)),'missing measurements')
    state=variant['state_sha256'];require(len(state)==32 and all(len(layer)==2 and all(len(h)==64 for h in layer) for layer in state),'missing full KV hashes')
    key=(tuple(live),chunk);identity=(state,variant['predictions'])
    if key in identities:require(identities[key]==identity,'state/predictions changed across cohorts')
    identities[key]=identity;allocated=None
    for sample in variant['measured']:
     require(sample['body_wall_seconds']>0 and sample['body_gpu_ms']>0,'invalid duration')
     memory=sample['memory'];n=memory['metal_allocated_bytes'];require(0<=n<=memory['metal_recommended_working_set_bytes']-256*1024*1024,'working-set bound exceeded')
     if allocated is None:allocated=n
     require(n==allocated,'sampled Metal allocation changed');measurements+=1
    medians[chunk]=statistics.median(s['body_wall_seconds'] for s in variant['measured'])
   require(variants[128]['predictions']==variants[wide]['predictions'],'prefill prediction divergence')
   rows.append(dict(cohort=cohort,live=live,narrow_chunk=128,wide_chunk=wide,narrow_median_body_seconds=medians[128],wide_median_body_seconds=medians[wide],body_ratio=medians[128]/medians[wide]))
  close=result['cleanup'];require(close['metal_allocated_bytes']<=4*1024*1024 and all(close[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')),'retained native owners')
 require(measurements==24,'incomplete paired coverage')
 read(a.study/'tools/manifest.json')
 for path,digest in read(a.study/'tools/manifest.json').items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'changed frozen tool')
 with a.output.open('x') as f:json.dump(dict(passed=True,measured_full_prefills=measurements,rows=rows,files=files,scope=__doc__),f,indent=2)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
