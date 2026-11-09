#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit actual-weight dense reconciliation range/state witnesses; CPU only."""
import argparse,copy,hashlib,json,math
from pathlib import Path

def require(ok,label):
 if not ok:raise ValueError(label)
def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def validate(data):
 require(data['completed'] is True,'incomplete')
 labels=['fresh','exact','append','changed','shorter','disjoint','near-capacity']
 require([r['label'] for r in data['records']]==labels,'missing/duplicate case')
 required={'empty','over-capacity','negative','bool','vocabulary','overflow-token','wrong-row','negative-row','stale-epoch','wrong-type','released-epoch','untracked-state','closed-owner'}
 require(len(data['rejected'])==13 and set(data['rejected'])==required,'missing rejection')
 for n,r in enumerate(data['records']):
  old=r['old'];q=r['query'];common=0
  while common<min(len(old),len(q)) and old[common]==q[common]:common+=1
  kept=common-1 if common==len(q) and common<len(old) else common
  require(r['common']==common and r['kept']==kept,'wrong common/replay cursor')
  stats=r['stats'];zero=(2*28*8*(len(old)-kept)*128+1024)*4 if kept<len(old) else 0
  require(stats['calls']==n+1 and stats['previous_tokens']==len(old) and stats['common_tokens']==common and stats['retained_tokens']==kept,'wrong native counters')
  require(stats['zeroed_bytes']==zero and stats['host_snapshot_bytes']==0 and stats['replay_last_token']==(kept<common),'wrong range clearing/replay')
  require(isinstance(stats['gpu_ms'],(int,float)) and math.isfinite(stats['gpu_ms']) and stats['gpu_ms']>=0,'invalid GPU duration')
  require(len(r['range_witness'])==28,'missing layer witness')
  for parts in r['range_witness']:
   require(len(parts)==2,'missing KV part')
   for part in parts:require(part['actual']==part['expected'] and all(digest(part[k]) for k in ('actual','expected','survivor')),'incorrect prefix/suffix/survivor')
  require(r['continued_state']==r['reference_state'] and len(r['continued_state'])==28 and all(len(x)==2 and all(digest(v) for v in x) for x in r['continued_state']),'continued state differs')
  require(r['prediction']==r['reference_prediction'] and len(r['prediction'])==2,'continued prediction differs')
 for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):require(data['cleanup'][key]==0,'live owner')
 require(0<=data['cleanup']['metal_allocated_bytes']<=4*1024*1024,'allocation after close')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();data=json.loads(a.result.read_text());validate(data);rejected=[]
 mutations={
 'incomplete':lambda d:d.update(completed=False),'missing_case':lambda d:d['records'].pop(),
 'bad_cursor':lambda d:d['records'][3].update(kept=5),'bad_range':lambda d:d['records'][3]['stats'].update(zeroed_bytes=0),
 'host_copy':lambda d:d['records'][3]['stats'].update(host_snapshot_bytes=1),'no_replay':lambda d:d['records'][4]['stats'].update(replay_last_token=False),
 'modified_prefix':lambda d:d['records'][3]['range_witness'][0][0].update(actual='0'*64),
 'bad_continuation':lambda d:d['records'][3]['continued_state'][1].__setitem__(0,'0'*64),
 'bad_prediction':lambda d:d['records'][3]['prediction'].__setitem__(0,-1),
 'missing_guard':lambda d:d['rejected'].pop(),'live_owner':lambda d:d['cleanup'].update(garnet_live_matrix_owners=1)}
 for label,mutate in mutations.items():
  changed=copy.deepcopy(data);mutate(changed)
  try:validate(changed)
  except (ValueError,KeyError,TypeError,IndexError):rejected.append(label)
  else:raise ValueError('corruption accepted: '+label)
 with a.output.open('x') as stream:json.dump(dict(completed=True,cases=7,rejected_corruptions=rejected,result_sha256=hashlib.sha256(a.result.read_bytes()).hexdigest(),auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope=__doc__),stream,indent=2)
 print('dense reconcile audit PASS',len(rejected),'corruptions rejected')
if __name__=='__main__':main()
