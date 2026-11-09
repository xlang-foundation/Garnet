#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify synchronous head GPU telemetry and unchanged native KV/hidden state."""
import argparse,copy,hashlib,json,math
from pathlib import Path
def need(v,m):
 if not v:raise ValueError(m)
def validate(old,new):
 need(old['completed'] is False and len(old['heads'])==1,'old failure not reproduced')
 h=old['heads'][0];need(h['gpu_ms']==old['body_gpu_ms'] and h['gpu_ms']>h['wall_ms']+1,'old failure is not stale body timing')
 need(new['completed'] is True and len(new['heads'])==4,'incomplete new probe')
 need(old['before']==new['before'],'prefill KV/hidden changed')
 state=new['before'];need(len(state['kv'])==32 and all(len(v)==2 for v in state['kv']),'state geometry')
 hashes=[h for layer in state['kv'] for h in layer]+[state['hidden']];need(all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in hashes),'state hashes')
 canonical=old['heads'][0]['ids'];need(len(canonical)==4,'head width')
 for x,(confidence,mask) in zip(new['heads'],[(False,[1,1,1,1]),(False,[1,0,0,0]),(False,[0,1,0,1]),(True,[1,1,1,1])]):
  need(x['confidence'] is confidence and x['mask']==mask,'head coverage')
  need(x['ids']==[canonical[i] if mask[i] else -1 for i in range(4)],'head token parity')
  need(x['state']==state,'head mutated KV/hidden')
  need(all(type(x[k]) in (int,float) and math.isfinite(x[k]) and x[k]>0 for k in ['gpu_ms','wall_ms']) and x['gpu_ms']<=x['wall_ms']+1,'GPU span exceeds synchronous wall')
 for d in [old,new]:
  c=d['cleanup'];need(c['metal_allocated_bytes']<=4*1024*1024 and all(c[k]==0 for k in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']),'native owner leak')
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;need(not a.output.exists(),'fresh output required');files={};records=[]
 def read(p):
  b=p.read_bytes();files[str(p.resolve())]=hashlib.sha256(b).hexdigest();return json.loads(b)
 for label,exitcode in [('old',1),('new',0)]:
  need(int((r/('head-'+label+'4.status')).read_text())==exitcode,'wrong process exit');cmd=read(r/('head-'+label+'-command4.json'));need(cmd['expected_exit']==exitcode,'wrong expected exit')
  for path,sha in [(Path(cmd['argv'][2]),cmd['script_sha256']),(Path(cmd['argv'][3])/'libgarnet.dylib',cmd['library_sha256'])]:
   actual=hashlib.sha256(path.read_bytes()).hexdigest();need(actual==sha,'source/library changed');files[str(path.resolve())]=actual
  records.append(read(r/('head-'+label+'4/results.json')))
 validate(*records);badcases=[]
 def reject(label,mutate):
  b=copy.deepcopy(records);mutate(b)
  try:validate(*b)
  except (ValueError,KeyError,TypeError):badcases.append(label)
  else:raise ValueError('accepted corruption '+label)
 reject('old-wrong-failure',lambda b:b[0]['heads'][0].update(gpu_ms=1))
 reject('stale-new-timer',lambda b:b[1]['heads'][0].update(gpu_ms=b[1]['body_gpu_ms']))
 reject('state-write',lambda b:b[1]['heads'][1]['state'].update(hidden='0'*64))
 reject('wrong-inactive-token',lambda b:b[1]['heads'][1]['ids'].__setitem__(1,0))
 reject('missing-confidence',lambda b:b[1]['heads'][-1].update(confidence=False))
 reject('owner-leak',lambda b:b[1]['cleanup'].update(garnet_live_matrix_owners=1))
 a.output.write_text(json.dumps(dict(completed=True,scope=__doc__,old_bug_reproduced=True,head_cases=4,exact_old_new_prefill_and_unchanged_head_state=True,heads=[{k:v for k,v in h.items() if k!='state'} for h in records[1]['heads']],rejected_corruptions=badcases,files=files,no_throughput_or_quality_claim=True),indent=2)+'\n');print('head audit PASS',len(badcases),'corruptions rejected')
if __name__=='__main__':main()
