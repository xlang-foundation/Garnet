#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit native B1 packed prefill A/B traces; descriptive native timings only."""
import argparse,copy,hashlib,json,math,statistics
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def finite_positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
def witness(w,slots):
    need(w['slots']==[slots] and len(w['state_sha256'])==32,'bad state witness geometry')
    hashes=[h for layer in w['state_sha256'] for h in layer]+[w['hidden_sha256']]
    need(all(len(layer)==2 for layer in w['state_sha256']) and len(hashes)==65 and all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in hashes),'bad state hashes')
def validate(fixture,records,inventories):
    need(len(records)==len(inventories)==4,'wrong cohort coverage');reference=None;results=[]
    need([len(c['prompt_ids']) for c in fixture['cases']]==[263,512,2048,2049],'changed fixture geometry')
    for record,inventory,mode in zip(records,inventories,['control','candidate','candidate','control']):
        need(record['completed'] is True and record['mode']==mode and len(record['cases'])==4,'incomplete collection')
        need('mistral_packed_prefill_optional' in record['backend'],'missing native capability')
        need(record['prepared']['graphs']==288 and record['prepared']['plans']==3,'wrong prepared plans')
        scratch=[r for r in inventory['references'] if r['role'].startswith('packed_prefill_')]
        roles={'packed_prefill_half_input':14680064}
        need(len(scratch)==1 and set(x['role'] for x in scratch)==set(roles) and len(set(x['buffer_id'] for x in scratch))==1,'wrong scratch ownership')
        for r in scratch:need(inventory['buffers'][r['buffer_id']]['length_bytes']==roles[r['role']],'wrong scratch allocation')
        for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:need(record['cleanup'][key]==0,'retained native owner')
        need(record['cleanup']['metal_allocated_bytes']<=4*1024*1024,'retained device buffers')
        current=[]
        for case,expected in zip(record['cases'],fixture['cases']):
            need(case['id']==expected['id'] and case['prompt_ids']==expected['prompt_ids'],'changed prompt IDs')
            n=len(case['prompt_ids']);ids=case['generated_ids'];need(1<=len(ids)<=128 and all(type(t) is int and 0<=t<32768 for t in ids),'bad generated IDs')
            need(len(ids)==128 or ids[-1]==2,'incorrect stopping');need(2 not in ids[:-1],'tokens after EOS')
            need(finite_positive(case['decode_seconds']),'bad decode time')
            witness(case['after_prefill'],n);witness(case['after_decode'],n+len(ids)-1)
            need(len(case['prefill'])==3,'wrong timing repetitions')
            lengths=[min(512,n-offset) for offset in range(0,n,512)]
            for sample in case['prefill']:
                need(finite_positive(sample['seconds']) and finite_positive(sample['gpu_ms']),'bad prefill timing')
                need([c['length'] for c in sample['chunks']]==lengths,'changed prefill schedule')
                need(all(c['packed_dispatches']==(160 if mode=='candidate' and c['length']>=128 else 0) and finite_positive(c['gpu_ms']) for c in sample['chunks']),'wrong native dispatch witness')
                need(abs(sum(c['gpu_ms'] for c in sample['chunks'])-sample['gpu_ms'])<1e-7,'wrong GPU time accounting')
            current.append(dict(id=case['id'],prompt_ids=case['prompt_ids'],generated_ids=ids,after_prefill=case['after_prefill'],after_decode=case['after_decode']))
        if reference is None:reference=current
        need(current==reference,'generated IDs or full KV/hidden witnesses differ')
        results.append(record)
    comparisons=[]
    for cohort,(ci,ni) in enumerate([(0,1),(3,2)],1):
        for c,n in zip(results[ci]['cases'],results[ni]['cases']):
            ct=statistics.median(x['seconds'] for x in c['prefill']);nt=statistics.median(x['seconds'] for x in n['prefill']);cg=statistics.median(x['gpu_ms'] for x in c['prefill']);ng=statistics.median(x['gpu_ms'] for x in n['prefill'])
            comparisons.append(dict(cohort=cohort,id=c['id'],prompt_tokens=len(c['prompt_ids']),control_prefill_seconds=ct,candidate_prefill_seconds=nt,prefill_ratio=ct/nt,control_gpu_ms=cg,candidate_gpu_ms=ng,gpu_ratio=cg/ng,generated_tokens=len(c['generated_ids'])))
    return comparisons

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--revision',type=int,choices=[1,2],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;need(not a.output.exists(),'fresh output required')
    fixture=json.loads((r/'fixture1.json').read_text());files=[r/'fixture1.json'];records=[];inventories=[]
    revision=str(a.revision);library_path=r/('library'+revision)/'libgarnet.dylib';library_sha=sha(library_path)
    for stem in ['cohort1-control','cohort1-candidate','cohort2-candidate','cohort2-control']:
        command=r/(stem+'-command'+revision+'.json');status=r/(stem+'-status'+revision+'.json');result=r/(stem+revision+'/results.json');inventory=r/(stem+revision+'/prepared-inventory.json')
        need(json.loads(status.read_text())['returncode']==0,'nonzero process exit');need(json.loads(command.read_text())['library_sha256']==library_sha,'changed library')
        files.extend([command,status,result,inventory]);records.append(json.loads(result.read_text()));inventories.append(json.loads(inventory.read_text()))
    for path,value in fixture['sources'].items():need(sha(Path(path))==value,'changed frozen workload')
    comparisons=validate(fixture,records,inventories);rejections=[]
    def reject(label,mutate):
        bad=copy.deepcopy(records);inv=copy.deepcopy(inventories);mutate(bad,inv)
        try:validate(fixture,bad,inv)
        except (ValueError,KeyError,IndexError,TypeError):rejections.append(label)
        else:raise ValueError('corruption accepted: '+label)
    reject('incomplete',lambda b,i:b[0].update(completed=False))
    reject('generated-id',lambda b,i:b[1]['cases'][0]['generated_ids'].__setitem__(0,(b[1]['cases'][0]['generated_ids'][0]+1)%32768))
    reject('kv-state',lambda b,i:b[2]['cases'][2]['after_decode']['state_sha256'][31].__setitem__(1,'0'*64))
    reject('hidden',lambda b,i:b[3]['cases'][1]['after_prefill'].update(hidden_sha256='1'*64))
    reject('slots',lambda b,i:b[0]['cases'][0]['after_prefill'].update(slots=[1]))
    reject('dispatch',lambda b,i:b[1]['cases'][1]['prefill'][0]['chunks'][0].update(packed_dispatches=159))
    reject('fallback-route',lambda b,i:b[1]['cases'][3]['prefill'][0]['chunks'][-1].update(packed_dispatches=160))
    reject('nan-time',lambda b,i:b[0]['cases'][0]['prefill'][0].update(seconds=float('nan')))
    reject('plan-growth',lambda b,i:b[0]['prepared'].update(plans=4))
    reject('owner-leak',lambda b,i:b[0]['cleanup'].update(garnet_live_matrix_owners=1))
    reject('missing-scratch',lambda b,i:i[0]['references'].pop(next(j for j,x in enumerate(i[0]['references']) if x['role'].startswith('packed_prefill_'))))
    hashes={str(p.resolve()):sha(p) for p in files};hashes[str(library_path.resolve())]=library_sha
    a.output.write_text(json.dumps(dict(completed=True,scope=__doc__,comparisons=comparisons,paired_sequences=16,generated_ids_compared=sum(len(c['generated_ids']) for x in records for c in x['cases']),full_layer_witnesses_per_boundary=32,exact_prefill_and_decode_state_parity=True,rejected_corruptions=rejections,files=hashes,no_http_or_quality_admission=True),indent=2)+'\n')
    print('model audit PASS',len(rejections),'corruptions rejected')
if __name__=='__main__':main()
