#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit same-session alternating full-prefill state and paired body timing."""
import argparse,hashlib,json,statistics
from pathlib import Path

def require(value,message):
 if not value:raise ValueError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files={}
 def read(path):
  data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
 status=read(a.study/'status.json');require(status['status']=='terminal' and status['returncode']==0,'unfinished native process')
 data=read(a.study/'result.json');require(data['completed'] is True and len(data['groups'])==3,'missing completed owner groups')
 require('verification_affine_route_control' in data['backend'] and 'chunk_flash_attention_fp32_operands' not in data['backend'] and 'chunk_attention_full_write_scratch' not in data['backend'],'wrong compiled experiment')
 pipeline=data['prepared_pipeline_count'];require(pipeline>0,'missing prepared pipelines');rows=[];measured=0
 for group,live in zip(data['groups'],[[0],[0,1],[0,1,2,3]]):
  require(group['live']==live and len(group['runs'])==10,'changed owner/run coverage')
  expected_order=[(pair,variant) for pair in range(-1,4) for variant in (['control','candidate'] if pair<0 or pair%2==0 else ['candidate','control'])]
  require([(r['pair'],r['variant']) for r in group['runs']]==expected_order,'changed balanced AB/BA order');identity=None;allocation=None;pairs={}
  for run in group['runs']:
   require(run['warmup']==(run['pair']<0) and run['pipeline_count']==pipeline,'warmup/pipeline drift')
   state=run['state_sha256'];require(len(state['KV'])==32 and all(len(x)==2 and all(len(h)==64 for h in x) for x in state['KV']) and len(state['hidden'])==64,'missing complete state hashes')
   value=(state,run['predictions'])
   if identity is None:identity=value
   require(identity==value,'control/candidate complete state or predictions differ')
   require(len(run['chunk_dispatches'])==16,'missing actual full-prefill chunks');candidate=run['variant']=='candidate'
   require(all(c==dict(reuse64=0 if candidate else 160,reuse128=160 if candidate else 0) for c in run['chunk_dispatches']),'wrong observed native projection route')
   memory=run['memory'];n=memory['metal_allocated_bytes'];require(0<=n<=memory['metal_recommended_working_set_bytes']-256*1024*1024,'working-set bound exceeded')
   if allocation is None:allocation=n
   require(n==allocation,'allocation changed within owner group');require(run['body_wall_seconds']>0 and run['body_gpu_ms']>0,'invalid duration')
   if not run['warmup']:pairs.setdefault(run['pair'],{})[run['variant']]=run;measured+=1
  require(set(pairs)==set(range(4)) and all(set(v)=={'control','candidate'} for v in pairs.values()),'missing paired variant')
  ratios=[pairs[i]['control']['body_wall_seconds']/pairs[i]['candidate']['body_wall_seconds'] for i in range(4)]
  gpu=[pairs[i]['control']['body_gpu_ms']/pairs[i]['candidate']['body_gpu_ms'] for i in range(4)]
  rows.append(dict(owners=len(live),pairs=4,control_median_seconds=statistics.median(pairs[i]['control']['body_wall_seconds'] for i in range(4)),candidate_median_seconds=statistics.median(pairs[i]['candidate']['body_wall_seconds'] for i in range(4)),paired_body_ratios=ratios,median_paired_body_ratio=statistics.median(ratios),paired_gpu_ratios=gpu,median_paired_gpu_ratio=statistics.median(gpu),sampled_allocated_bytes=allocation))
 require(measured==24,'missing measured full prefills');close=data['cleanup'];require(close['metal_allocated_bytes']<=4*1024*1024 and all(close[k]==0 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')),'retained native owners')
 for path,digest in read(a.study/'tools/manifest.json').items():require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'changed frozen tool')
 result=dict(passed=True,complete_same_session_state_exact=True,measured_full_prefills=measured,warmup_full_prefills=6,rows=rows,files=files,scope=__doc__)
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
