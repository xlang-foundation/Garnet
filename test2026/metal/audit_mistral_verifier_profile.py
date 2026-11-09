# SPDX-License-Identifier: Apache-2.0
"""Audit raw native encoder samples against the full saved-input verifier replay."""
import json,sys,math,collections,hashlib
from pathlib import Path

def require(condition,message):
 if not condition:raise ValueError(message)
def audit(samples,replay,original):
 require(replay['completed'] and original['completed'],'incomplete run')
 require(len(replay['records'])==len(replay['cases'])>0,'missing cases')
 require(len(set(replay['cases']))==len(replay['cases']),'duplicate case')
 require(replay['cases']==[c['id'] for c in original['inputs']['cases']],'missing/reordered workload case')
 require([r['case_id'] for r in replay['records']]==replay['cases'],'record/case mismatch')
 expected=[];tokens=0
 for row in replay['records']:
  old=next(x for x in original['records'] if x['case_id']==row['case_id'] and x['mode']=='resident')
  require(row['generated_ids']==old['generated_ids'],'target IDs changed')
  require(row['state_sha256']==old['state_sha256'] and row['hidden_sha256']==old['hidden_sha256'],'target state changed')
  require(len(row['rounds'])==len(old['rounds']),'round count changed')
  for r,o in zip(row['rounds'],old['rounds']):
   for k in ('slot','consumed','submitted_ids','emitted_ids'):require(r[k]==o[k],'replay inputs/outputs changed')
   if r['metrics'] is not None:
    require(r['metrics']['target_affine_dispatches']==225 and r['metrics']['target_affine_misses']==0,'unprepared target route')
    expected.append(r['metrics']['target_gpu_ms'])
  require(row['generated_ids']==[row['generated_ids'][0]]+[t for r in row['rounds'] for t in r['emitted_ids']],'incomplete emitted sequence')
  tokens+=len(row['generated_ids'])
 for k in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
  require(replay['cleanup'][k]==0,'live native owners')
 require(replay['cleanup']['metal_allocated_bytes']<=4*1024*1024,'cleanup bytes')
 selected=[];groups=collections.defaultdict(lambda:dict(count=0,ms=0.0));union_total=0.;sum_total=0.;overlap_total=0.
 for row in samples:
  labels=collections.Counter(e['label'] for e in row['encoders'])
  if not labels['EncodeStateKernel/22']:continue
  require(row['command_success'] and row['complete_compute_sampling'],'incomplete sampling')
  require(row['invalid_encoders']==0 and row['dropped_encoders']==0,'invalid/dropped samples')
  require(labels['EncodeStateKernel/22']==32 and labels['EncodeStateKernel/23']==32,'attention dispatch count')
  require(labels['EncodeStateKernel/26']==64,'KV backup dispatch count')
  require(sum(n for label,n in labels.items() if label.startswith('EncodeAffineInt4'))==225,'projection dispatch count')
  c=row['calibration'];require(c['valid'],'uncalibrated timestamps')
  require(c['cpu_end_ns']>c['cpu_start_ns'] and c['gpu_end']>c['gpu_start'],'invalid calibration span')
  scale=(c['cpu_end_ns']-c['cpu_start_ns'])/(c['gpu_end']-c['gpu_start'])
  require(math.isfinite(scale) and math.isclose(scale,c['ns_per_counter_tick'],rel_tol=1e-12),'wrong timestamp units')
  intervals=[];total=0.
  for e in row['encoders']:
   require(e['valid'] and c['gpu_start']<=e['start_ticks']<=e['end_ticks']<=c['gpu_end'],'invalid timestamp range')
   ms=(e['end_ticks']-e['start_ticks'])*scale/1e6
   require(math.isclose(ms,e['ms'],rel_tol=1e-12,abs_tol=1e-12),'wrong timestamp duration')
   intervals.append((e['start_ticks'],e['end_ticks']));groups[e['label']]['count']+=1;groups[e['label']]['ms']+=ms;total+=ms
  intervals.sort();start,end=intervals[0];ticks=0
  for a,b in intervals[1:]:
   if a>end:ticks+=end-start;start,end=a,b
   else:end=max(end,b)
  ticks+=end-start;union=ticks*scale/1e6
  span=(max(b for a,b in intervals)-intervals[0][0])*scale/1e6
  # Counter conversion must agree with independent command-buffer timestamps.
  require(0<union<=row['gpu_ms']+0.03 and span<=row['gpu_ms']+0.03,'counter/command clocks disagree')
  require(union>=row['gpu_ms']*0.5,'unexplained large sampling gap')
  union_total+=union;sum_total+=total;overlap_total+=max(0.,total-union);selected.append(row)
 require(len(selected)==len(expected)>0,'missing verifier samples')
 require(len({(r['context'],r['sequence']) for r in samples})==len(samples),'duplicate command identity')
 for sample,gpu in zip(selected,expected):require(sample['gpu_ms']==gpu,'sample/replay command mismatch')
 gpu=sum(expected)
 result=dict(passed=True,scope=__doc__,cases=len(replay['records']),target_ids=tokens,verifier_calls=len(expected),
  target_gpu_ms=gpu,encoder_union_ms=union_total,encoder_sum_ms=sum_total,overlap_ms=overlap_total,
  other_command_ms=gpu-union_total,groups=[dict(label=k,**v,mean_ms=v['ms']/v['count'],percent_command_gpu=100*v['ms']/gpu) for k,v in sorted(groups.items(),key=lambda kv:-kv[1]['ms'])],
  caveat='Encoder duration sums can overlap; union is the observed compute coverage. Instrumented replay excludes draft inference and is not HTTP throughput or a speed qualification.')
 return result
if __name__=='__main__':
 samples=[json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines()]
 replay=json.loads(Path(sys.argv[2]).read_text());original=json.loads(Path(sys.argv[3]).read_text())
 result=audit(samples,replay,original)
 Path(sys.argv[4]).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='groups'}));print(json.dumps(result['groups'][:8],indent=2))
