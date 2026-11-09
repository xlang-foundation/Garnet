#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit paired actual-model stage evidence; no HTTP/comparator speed claim."""
import argparse,hashlib,json,statistics
from pathlib import Path

def need(value,message):
 if not value:raise ValueError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files={};data={};pids=set();last=0;observations=0
 def read(path):
  raw=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 golden=read(a.control/'merge-closed-intervals/request-0.result.json')['generated_ids'][:32];need(len(golden)==32,'short control')
 for path,digest in read(a.study/'tools/manifest.json').items():need(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'changed frozen model tool')
 for batch in [1,4]:
  for cohort,variant in [(1,'control'),(2,'candidate'),(3,'candidate'),(4,'control')]:
   status=read(a.study/f'b{batch}-c{cohort}-status.json');need(status['status']=='terminal' and status['returncode']==0 and status['batch']==batch and status['cohort']==cohort and status['variant']==variant,'unfinished/wrong model cohort');need(status['pid'] not in pids and last<=status['started_ns']<status['finished_ns'],'overlapping/repeated native process');pids.add(status['pid']);last=status['finished_ns']
   result=read(a.study/f'b{batch}-c{cohort}.json');need(result['completed'] and result['batch']==batch and result['steps']==32 and result['chunk']==128 and result['case_id']=='merge-closed-intervals' and len(result['records'])==3,'changed model coverage')
   need(('affine_tiled_decode_fp32_ordered32' in result['backend'])==(variant=='candidate'),'wrong native representation');decode=[];prefill=[]
   for index,record in enumerate(result['records']):
    need(record['repetition']==index and record['generated_rows']==[golden for _ in range(batch)],'changed complete generated token prefix');observations+=32*batch;need(len(record['decode_samples'])==31,'missing decode frames');duration=sum(s['step_wall_ms'] for s in record['decode_samples']);need(duration>0 and all(s['gpu_ms']>0 for s in record['decode_samples']),'invalid GPU/decode duration');memory=record['memory'];need(memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-256*1024*1024,'workspace exceeded')
    if index>0:decode.append(duration);prefill.append(record['prefill']['wall_seconds'])
   close=result['cleanup'];need(close['metal_allocated_bytes']<=4*1024*1024 and all(close[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'retained native owners');data[(batch,cohort)]=dict(decode_ms=statistics.median(decode),prefill_seconds=statistics.median(prefill),aggregate_decode_tps=batch*31/(statistics.median(decode)/1000))
 rows=[]
 for batch in [1,4]:
  for old,new in [(1,2),(4,3)]:
   control,candidate=data[(batch,old)],data[(batch,new)];rows.append(dict(batch=batch,control_cohort=old,candidate_cohort=new,control=control,candidate=candidate,decode_ratio=control['decode_ms']/candidate['decode_ms']))
 need(observations==1920,'missing token observations')
 with a.output.open('x') as f:json.dump(dict(passed=True,rows=rows,sequences=24,measured_sequences=16,exact_token_observations=observations,files=files,scope=__doc__,limits='One84-token coding prompt,32 generated IDs, repeated identical rows, first repetition excluded; not HTTP, long-context, new coding-quality or2x admission.'),f,indent=2)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
