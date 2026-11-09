# SPDX-License-Identifier: Apache-2.0
"""Audit the serial Mistral streamed-register experiment; ratios are operator-only."""
import json,math,hashlib,statistics,sys
from pathlib import Path

def need(x,message):
 if not x:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(root):
 root=Path(root).resolve();manifest=json.loads((root/'operator-manifest1.json').read_text())
 for name,digest in manifest['files'].items():need(sha(Path(name))==digest,'frozen source/binary changed')
 collector=json.loads((root/'collector-status1.json').read_text())
 need(collector['completed'] and len(collector['runs'])==4,'incomplete collection')
 shapes=[(1,64),(7,192),(17,64),(33,128),(4096,4096),(1024,4096),(14336,4096),(4096,14336),(32768,4096)]
 expected=[]
 for n,k in shapes:
  expected += [(n,k,r,False,[]) for r in (1,2,3,4)]
  expected += [(n,k,4,True,[r for r in range(4) if mask&(1<<r)]) for mask in range(16)]
 previous=manifest['frozen_before_run_ns'];pids=set();baseline={};data={};words=0;fp64=0;outputs={}
 for run,identity in zip(collector['runs'],[(1,'control'),(1,'stream'),(2,'stream'),(2,'control')]):
  cohort,label=identity
  need((run['cohort'],run['label'])==identity and run['returncode']==0,'run failed/order changed')
  need(run['started_ns']>=previous and run['finished_ns']>run['started_ns'] and run['pid'] not in pids,'overlapping/repeated process');previous=run['finished_ns'];pids.add(run['pid'])
  binary=root/('operator-'+label);output=root/f'cohort{cohort}-{label}-outputs';raw=root/f'cohort{cohort}-{label}.jsonl'
  need(run['argv']==[str(binary),str(output),'9'] and run['binary_sha256']==sha(binary) and run['raw_sha256']==sha(raw),'changed launch/output')
  rows=[json.loads(line) for line in raw.read_text().splitlines()]
  need(len(rows)==182 and rows[-1]==dict(kind='closed',cells=180,owners=0),'incomplete native outputs')
  m=rows[0];need(m['device']=='Apple M5' and m['warmups']==3 and m['repetitions']==9 and m['maximum_rows']==4 and m['profile_is_upper_bound'] is True and m['coefficient_dtype']=='FP16','profile/precision changed')
  for index,(cell,(n,k,r,indexed,selected)) in enumerate(zip(rows[1:-1],expected)):
   need(all([cell['kind']=='cell',cell['index']==index,cell['outputs']==n,cell['inner']==k,cell['rows']==r,cell['indexed']==indexed,cell['selected']==selected,cell['words']==r*n]),'cell shape/mask changed')
   active=len(selected) if indexed else r
   need(cell['fp64_samples']==3*active and math.isfinite(cell['maximum_error_fraction']) and 0<=cell['maximum_error_fraction']<=1,'FP64 dot bound failed');fp64+=cell['fp64_samples']
   for key in ('gpu_ms','wall_ms'):need(len(cell[key])==9 and all(math.isfinite(t) and (t>0 if active else t>=0) for t in cell[key]),'invalid time sample')
   p=output/f'case-{index}.bin';need(p.stat().st_size==4*r*n,'incomplete binary');digest=sha(p);outputs[str(p)]=digest
   if identity==(1,'control'):baseline[index]=digest
   else:need(digest==baseline[index],'complete output differs');words+=r*n
  data[identity]=rows[1:-1]
 comparisons=[]
 for cohort in (1,2):
  for i,(n,k,r,indexed,selected) in enumerate(expected):
   if (len(selected) if indexed else r)==0:continue
   ctrl=data[cohort,'control'][i];cand=data[cohort,'stream'][i]
   comparisons.append(dict(cohort=cohort,index=i,outputs=n,inner=k,rows=r,indexed=indexed,selected=selected,control_gpu_ms=statistics.median(ctrl['gpu_ms']),stream_gpu_ms=statistics.median(cand['gpu_ms']),gpu_ratio=statistics.median(ctrl['gpu_ms'])/statistics.median(cand['gpu_ms']),wall_ratio=statistics.median(ctrl['wall_ms'])/statistics.median(cand['wall_ms'])))
 return dict(passed=True,processes=4,cells_per_process=180,compared_f32_words=words,fp64_samples=fp64,comparisons=comparisons,output_sha256=outputs,scope=__doc__,whole_model_qualified=False)
if __name__=='__main__':
 result=audit(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ('passed','processes','compared_f32_words','fp64_samples')}))
 for row in result['comparisons']:
  if row['rows']==4 and not row['indexed'] and row['inner']>1000:print(json.dumps(row))
