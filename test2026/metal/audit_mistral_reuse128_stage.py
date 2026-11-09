#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit opposite-order cooperative projection body measurements; no serving admission."""
import argparse,hashlib,json,statistics
from pathlib import Path

def require(v,message):
 if not v:raise ValueError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files={};cohorts={};identities={};sample_count=0
 def read(path):
  data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
 for cohort,variant in [(1,'control'),(2,'candidate'),(3,'candidate'),(4,'control')]:
  status=read(a.study/f'cohort-{cohort}-status.json');result=read(a.study/f'cohort-{cohort}.json')
  require(status['status']=='terminal' and status['returncode']==0 and status['variant']==variant,'unfinished/wrong cohort')
  require(result['completed'] is True and result['variant']==variant and len(result['groups'])==2,'incomplete body coverage')
  require(('mpp_affine_reuse128_optional' in result['backend'])==(variant=='candidate'),'wrong compiled projection variant')
  groups={}
  for group,live in zip(result['groups'],[[0],[0,1,2,3]]):
   require(group['live']==live and len(group['measured'])==3 and [s['repeat'] for s in group['measured']]==[0,1,2],'changed owner/sample coverage')
   require(group['dispatches_per_chunk']==dict(reuse64=0 if variant=='candidate' else 160,reuse128=160 if variant=='candidate' else 0),'wrong observed projection dispatch route')
   state=group['state_sha256'];require(len(state['KV'])==32 and all(len(x)==2 and all(len(h)==64 for h in x) for x in state['KV']) and len(state['hidden'])==64,'missing complete state hashes')
   identity=(state,group['predictions']);key=tuple(live)
   if key in identities:require(identities[key]==identity,'old/new complete KV, hidden or predictions differ')
   identities[key]=identity;allocated=None
   for sample in group['measured']:
    require(sample['body_wall_seconds']>0 and sample['body_gpu_ms']>0,'invalid body duration');memory=sample['memory'];n=memory['metal_allocated_bytes'];require(0<=n<=memory['metal_recommended_working_set_bytes']-256*1024*1024,'working-set bound exceeded')
    if allocated is None:allocated=n
    require(n==allocated,'sampled allocation changed');sample_count+=1
   groups[len(live)]=dict(body_wall_seconds=statistics.median(s['body_wall_seconds'] for s in group['measured']),body_gpu_ms=statistics.median(s['body_gpu_ms'] for s in group['measured']),sampled_allocated_bytes=allocated)
  close=result['cleanup'];require(close['metal_allocated_bytes']<=4*1024*1024 and all(close[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')),'retained native owners')
  cohorts[cohort]=dict(variant=variant,groups=groups,prepared_bytes=result['prepared']['metal_allocated_bytes'])
 rows=[]
 for control,candidate in [(1,2),(4,3)]:
  for owners in [1,4]:
   old,new=cohorts[control]['groups'][owners],cohorts[candidate]['groups'][owners]
   rows.append(dict(control_cohort=control,candidate_cohort=candidate,owners=owners,control_median_body_seconds=old['body_wall_seconds'],candidate_median_body_seconds=new['body_wall_seconds'],body_ratio=old['body_wall_seconds']/new['body_wall_seconds'],prepared_bytes_saved=cohorts[control]['prepared_bytes']-cohorts[candidate]['prepared_bytes']))
 require(sample_count==24,'missing full-prefill measurements')
 for path,digest in read(a.study/'tools/manifest.json').items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'changed frozen tool')
 with a.output.open('x') as f:json.dump(dict(passed=True,measured_full_prefills=sample_count,rows=rows,cohorts=cohorts,complete_old_new_state_exact=True,files=files,scope=__doc__),f,indent=2)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
