#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check complete private-cache bytes, CPU oracle, serial processes and GPU medians."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def expected(scenario):
    batch,heads,dim,capacity,chunk=4,8,128,4096,128
    i=np.arange(batch*heads*capacity*dim,dtype=np.int64)
    cache=(((i*131+71)%65521-32760).astype('<f4')*np.float32(2**-16)).reshape(batch,heads,capacity,dim)
    i=np.arange(batch*chunk*heads*dim,dtype=np.int64)
    tokens=(((i*37+97)%65521-32760).astype('<f4')*np.float32(2**-15)).reshape(batch,chunk,heads,dim)
    for row in range(batch):
        owner=row==0 if scenario==1 else row==batch-1 if scenario==2 else row%2==0 if scenario==3 else False if scenario==5 else True
        for t in range(chunk):
            active=owner and (scenario!=4 or t+1<chunk) and (scenario!=6 or t%3!=1)
            if active:cache[row,:,t%5 if scenario==6 else capacity-chunk+t,:]=tokens[row,t]
    return hashlib.sha256(cache.tobytes()).hexdigest()


def audit(root,draft):
    root,draft=root.resolve(),draft.resolve();files={}
    def sha(p):
        with p.open('rb') as f:files[str(p)]=hashlib.file_digest(f,'sha256').hexdigest()
        return files[str(p)]
    def read(p):sha(p);return json.loads(p.read_text())
    collection=read(root/'collector-status.json');require(collection['completed'] is True and len(collection['runs'])==4,'incomplete KV campaign')
    previous,pids,data,total=0,set(),{},0
    oracle={scenario:expected(scenario) for scenario in range(9)}
    for run,identity in zip(collection['runs'],[(0,'control'),(0,'candidate'),(1,'candidate'),(1,'control')]):
        cohort,variant=identity;label=f'cohort{cohort}-{variant}'
        require((run['cohort'],run['variant'])==identity and run['returncode']==0 and type(run['pid']) is int
                and run['pid']>0 and run['pid'] not in pids and previous<=run['started_ns']<run['finished_ns'],'failed/overlapping native process')
        previous=run['finished_ns'];pids.add(run['pid']);binary=Path(run['argv'][0])
        require(sha(binary)==run['binary_sha256'] and run['argv'][1:]==['--benchmark',str(root/label)],'changed native benchmark contract')
        log=root/(label+'.jsonl');sha(log);rows=[json.loads(l) for l in log.read_text().splitlines()]
        require(len(rows)==11 and rows[0]['kind']=='metadata' and rows[0]['device']=='Apple M5'
                and rows[0]['scatter_compiled']==(variant=='candidate')
                and rows[-1]==dict(kind='closed',cases=9,owners=0),'incomplete/mismatched native cases')
        for scenario,cell in enumerate(rows[1:-1]):
            require(cell['kind']=='cell' and cell['index']==cell['scenario']==scenario and
                    (cell['batch'],cell['heads'],cell['dim'],cell['capacity'],cell['chunk'])==(4,8,128,4096,128)
                    and cell['inplace']==(scenario!=7) and cell['contiguous']==(scenario not in (6,8))
                    and cell['full_cpu_reference_words']==4*8*128*4096 and cell['warmups']==3,'wrong KV shape/oracle coverage')
            require(cell['scatter_dispatches']==int(variant=='candidate' and scenario not in (6,7,8)),'wrong scatter/copy/scan route')
            for field in ('gpu_ms','wall_ms'):require(len(cell[field])==9 and all(math.isfinite(v) and v>0 for v in cell[field]),'invalid native duration')
            p=root/label/f'case-{scenario}.f32';require(p.stat().st_size==cell['full_cpu_reference_words']*4 and sha(p)==oracle[scenario],'complete KV differs from independent CPU oracle')
            total+=cell['full_cpu_reference_words']
        data[identity]=rows[1:-1]
    comparisons=[]
    for cohort in (0,1):
        for scenario in range(9):
            a,b=data[cohort,'control'][scenario],data[cohort,'candidate'][scenario]
            left,right=statistics.median(a['gpu_ms']),statistics.median(b['gpu_ms'])
            comparisons.append(dict(cohort=cohort,scenario=scenario,control_gpu_ms=left,candidate_gpu_ms=right,operator_speedup=left/right))
    src=root.parent/'mistral-kv-scatter-source3';manifest=read(draft/'source3-manifest.json')
    for name,digest in manifest.items():require(sha(src/name)==digest,'compiled source changed')
    sha(draft/'control-link3-command.json');sha(root.parent/'native-mistral-decode-reuse-build3/libgarnet_metal.a')
    sha(Path(__file__).resolve())
    return dict(passed=True,cases=36,full_cache_oracle_words=total,candidate_control_compared_words=total//2,
        comparisons=comparisons,files=files,scope=__doc__,model_qualified=False,speed_targets_achieved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','draft','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report=audit(a.collection,a.draft)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','cases','full_cache_oracle_words')}))
