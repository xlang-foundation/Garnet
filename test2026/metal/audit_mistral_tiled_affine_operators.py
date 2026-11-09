#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit complete native affine words and opposite-order operator timing only."""
import argparse,json,hashlib,statistics
from pathlib import Path

def need(v,message):
 if not v:raise ValueError(message)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();files={};results={};word_count=0;seen_pids=set();last_finish=0
 def read(path):
  data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return data
 for path,digest in json.loads(read(a.study/'tools/manifest.json')).items():need(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'changed frozen operator tool')
 read(Path(__file__))
 for mode,total in [('prefill',40),('decode',180)]:
  reference=None
  for cohort,variant in [(1,'control'),(2,'candidate'),(3,'candidate'),(4,'control')]:
   root=a.study/(mode+'-'+str(cohort));status=json.loads(read(root/'status.json'));need(status['status']=='terminal' and status['returncode']==0 and status['variant']==variant and status['mode']==mode,'unfinished/wrong operator process');need(status['cohort']==cohort and status['pid'] not in seen_pids and last_finish<=status['started_ns']<status['finished_ns'],'overlapping/duplicated process provenance');seen_pids.add(status['pid']);last_finish=status['finished_ns']
   records=[json.loads(line) for line in read(root/'run.log').decode().splitlines() if line.startswith('{')];need(records[0]['kind']=='metadata' and records[-1]['owners']==0,'missing metadata/cleanup')
   cells=[x for x in records if x['kind']=='cell'];need(len(cells)==total,'changed shape/cell coverage');words={};timings={}
   for cell in cells:
    if mode=='prefill':
     key=(cell['logical_rows'],cell['outputs'],cell['inner']);path=Path(cell['output_file']);need(path.parent.resolve()==(root/'outputs').resolve(),'unbound output file');need(cell['mode']=='indexed' and cell['profile_misses']==0 and cell['runtime_pipeline_compiles']==0 and cell['fp64_max_error_fraction']<=1,'prefill route/oracle failure');need(cell['reuse64_dispatches']==int(cell['logical_rows']>=64 and cell['outputs']>=4096),'not the selected64-row control route');count=cell['output_words']
    else:
     key=(cell['index'],cell['rows'],cell['outputs'],cell['inner'],cell['indexed'],tuple(cell['selected']));path=root/'outputs'/('case-'+str(cell['index'])+'.bin');need(cell['maximum_error_fraction']<=1,'decode oracle failed');count=cell['words']
    need(key not in timings,'duplicate operator cell');payload=read(path);need(len(payload)==4*count,'wrong complete output length');words[key]=payload;need(len(cell['gpu_ms'])==9 and len(cell['wall_ms'])==9 and all(v>0 for v in cell['gpu_ms']),'wrong/invalid timed samples');timings[key]=dict(gpu_ms=statistics.median(cell['gpu_ms']),wall_ms=statistics.median(cell['wall_ms']))
   if mode=='prefill':
    expected={(m,n,k) for m in [4,16,32,63,64,65,127,128,256,512] for n,k in [(1024,4096),(4096,4096),(14336,4096),(4096,14336)]}
   else:
    expected=set();index=0
    for n,k in [(1,64),(7,192),(17,64),(33,128),(4096,4096),(1024,4096),(14336,4096),(4096,14336),(32768,4096)]:
     for m in [1,2,3,4]:expected.add((index,m,n,k,False,()));index+=1
     for mask in range(16):expected.add((index,4,n,k,True,tuple(i for i in range(4) if mask&(1<<i))));index+=1
   need(set(words)==expected,'changed canonical operator coverage')
   if reference is None:reference=words
   else:
    need(set(words)==set(reference),'different operator identities')
    for key in reference:need(words[key]==reference[key],'old/new complete operator bytes differ');word_count+=len(words[key])//4
   results[(mode,cohort)]=timings
  del reference
 rows=[]
 for mode in ['prefill','decode']:
  for old,new in [(1,2),(4,3)]:
   for key,control in results[(mode,old)].items():
    candidate=results[(mode,new)][key];rows.append(dict(mode=mode,control_cohort=old,candidate_cohort=new,cell=key,control_gpu_ms=control['gpu_ms'],candidate_gpu_ms=candidate['gpu_ms'],gpu_ratio=control['gpu_ms']/candidate['gpu_ms']))
 with a.output.open('x') as f:json.dump(dict(passed=True,opposite_order_rows=rows,complete_compared_words=word_count,files=files,scope=__doc__,limitation='Separate serial processes with synthetic weights; no actual-model, serving or2x admission.'),f,indent=2)
 print('operator_audit_pass',len(rows),'rows',word_count,'complete compared words')
if __name__=='__main__':main()
