#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit indexed occupancy experiments, including poisoned write coverage.

This cannot qualify model speed, target8 verification or coding quality.
"""
import argparse,copy,hashlib,json,math,statistics,struct
from pathlib import Path
POISON=0x7fc0abcd
SHAPES=[(1,64),(7,192),(17,64),(33,128),(4096,4096),(1024,4096),(14336,4096),(4096,14336),(32768,4096)]

def need(v,msg):
 if not v:raise ValueError(msg)

def validate(records,payloads):
 meta=records[0];closed=records[-1];cells=records[1:-1]
 need(meta['kind']=='metadata' and meta['negative'] is False and meta['poison']==POISON and meta['warmups']==3 and meta['reps']==15,'metadata/poison')
 need(closed['kind']=='closed' and closed['cases']==45 and len(cells)==360,'incomplete')
 need(closed['checked_physical_words']==194842368 and closed['independent_fp64_outputs']==1014804,'coverage counts')
 rows=[];retained=0
 for index,(n,k,count) in enumerate((n,k,c) for n,k in SHAPES for c in (1,2,3,4,8)):
  physical=8 if count==8 else 4
  indices={1:[3],2:[3,0],3:[3,0,2],4:[3,0,2,1],8:[7,0,6,1,5,2,4,3]}[count]
  selected=set(indices)
  for swept in (False,True):
   group=cells[index*8+int(swept)*4:index*8+int(swept)*4+4]
   need([x['variant'] for x in group]==['control','c2','c4','c8'],'variant ordering')
   control=payloads[group[0]['output']]
   ms={}
   for cell in group:
    need(cell['kind']=='cell' and cell['index']==index and cell['outputs']==n and cell['inner']==k and cell['rows']==count and cell['physical_rows']==physical and cell['indices']==indices and cell['swept']==swept,'cell identity')
    wanted=f"case-{index}-{'swept' if swept else 'hot'}-{cell['variant']}.bin"
    need(cell['output']==wanted,'output binding')
    need(math.isfinite(cell['max_error_fraction']) and 0<=cell['max_error_fraction']<=1,'FP64 bound')
    for key in ('gpu_ms','wall_ms'):need(len(cell[key])==15 and all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0 for x in cell[key]),'samples')
    payload=payloads[wanted];need(len(payload)==physical*n*4,'truncated output')
    for i,(bits,) in enumerate(struct.iter_unpack('<I',payload)):
     if i//n in selected:
      need(math.isfinite(struct.unpack('<f',struct.pack('<I',bits))[0]),'missing active write')
     else:need(bits==POISON,'inactive overwritten')
    need(payload==control,'full retained native control bytes differ');retained+=physical*n
    ms[cell['variant']]=statistics.median(cell['gpu_ms'])
   rows.append(dict(outputs=n,inner=k,rows=count,swept=swept,gpu_median_ms=ms,c2_ratio=ms['control']/ms['c2'],c4_ratio=ms['control']/ms['c4'],c8_ratio=ms['control']/ms['c8']))
 return rows,retained

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--cohorts',type=int,nargs='+',default=[1,2]);a=p.parse_args();r=a.root;cohorts=[];rejected=[]
 need(len(a.cohorts)==len(set(a.cohorts)) and all(n in (1,2,3) for n in a.cohorts),'cohort selection')
 for number in a.cohorts:
  manifest=json.loads((r/f'manifest{number}.json').read_text())
  for name,h in manifest['files'].items():need(hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,'changed frozen source/binary')
  status=json.loads((r/f'status{number}.json').read_text());need(status['terminal'] and status['exit_code']==0,'nonterminal/failed collection')
  records=[json.loads(l) for l in (r/f'operator{number}.jsonl').read_text().splitlines()]
  payloads={c['output']:(r/f'outputs{number}'/c['output']).read_bytes() for c in records if c['kind']=='cell'}
  summary,words=validate(records,payloads)
  mutations={'incomplete':lambda d:d.pop(),'wrong_poison':lambda d:d[0].update(poison=0),'wrong_rows':lambda d:d[1].update(rows=8),'wrong_map':lambda d:d[1].update(indices=[0]),'bound':lambda d:d[1].update(max_error_fraction=2.),'nan_time':lambda d:d[1]['gpu_ms'].__setitem__(0,float('nan')),'wrong_coverage':lambda d:d[-1].update(checked_physical_words=1)}
  for label,fn in mutations.items():
   changed=copy.deepcopy(records);fn(changed)
   try:validate(changed,payloads)
   except (ValueError,KeyError,IndexError,TypeError):rejected.append([number,label])
   else:raise ValueError('corrupted records accepted: '+label)
  name=records[2]['output']
  for label,data in [('short_bytes',payloads[name][:-4]),('unwritten_active',payloads[name][:-4]+struct.pack('<I',POISON)),('overwritten_inactive',struct.pack('<I',0)+payloads[name][4:])]:
   changed=dict(payloads);changed[name]=data
   try:validate(records,changed)
   except (ValueError,KeyError,IndexError,TypeError):rejected.append([number,label])
   else:raise ValueError('corrupted binary accepted: '+label)
  cohorts.append(dict(number=number,scope='Exploratory max8 template for1..4 in cohort1; exact original max4 template in cohorts2/3. Cohorts2/3 share exact source and binary. All have true max8.',rows=summary,retained_physical_words=words,raw_sha256=hashlib.sha256((r/f'operator{number}.jsonl').read_bytes()).hexdigest()))
 for number in (2,3):
  negative=json.loads((r/f'negative-status{number}.json').read_text());need(negative['passed_expected_rejection'] and negative['exit_code']!=0 and 'Active output not written after poison' in negative['stderr'],'broken writer not rejected')
 if 2 in a.cohorts and 3 in a.cohorts:need(json.loads((r/'manifest2.json').read_text())['files']==json.loads((r/'manifest3.json').read_text())['files'],'cohort2/3 binary/source differ')
 with a.output.open('x') as f:json.dump(dict(completed=True,scope=__doc__,cohorts=cohorts,rejected_corruptions=rejected,known_bad_writer_rejected_twice=True,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,indent=2)
 print('occupancy audit PASS cohorts='+str(len(cohorts))+' cells='+str(360*len(cohorts))+' corruptions='+str(len(rejected)))
if __name__=='__main__':main()
