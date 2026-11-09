#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independently audit frozen-prompt native B4 prefill/generation traces."""
import argparse,copy,hashlib,json,math,statistics
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
def witness(w,slots):
    need(w['slots']==slots and len(w['state_sha256'])==32,'state geometry/slots')
    hashes=[h for pair in w['state_sha256'] for h in pair]+[w['hidden_sha256']]
    need(all(len(pair)==2 for pair in w['state_sha256']) and len(hashes)==65 and all(type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in hashes),'state hash geometry')
def validate(fixture,records,inventories):
    need(len(records)==len(inventories)==4,'four serial processes required');reference=None;comparisons=[];tokens=0
    for report,inventory,mode in zip(records,inventories,['control','candidate','candidate','control']):
        need(report['completed'] is True and report['mode']==mode and len(report['cases'])==2,'incomplete/wrong mode')
        need('mistral_packed_prefill_optional' in report['backend'],'wrong native backend')
        need(report['prepared']['graphs']==224 and report['prepared']['plans']==3,'wrong prepared profiles')
        scratch=[x for x in inventory['references'] if x['role'].startswith('packed_prefill_')];need(len(scratch)==1 and len(set(x['buffer_id'] for x in scratch))==1,'wrong scratch owners')
        need(sum(inventory['buffers'][x['buffer_id']]['length_bytes'] for x in scratch)==14680064,'wrong scratch budget')
        for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:need(report['cleanup'][key]==0,'native owner leak')
        need(report['cleanup']['metal_allocated_bytes']<=4*1024*1024,'device retention')
        current=[]
        for c,expected in zip(report['cases'],fixture['cases']):
            need(c['id']==expected['id'] and c['prompt_ids']==expected['prompts'],'changed frozen IDs')
            lengths=[len(p) for p in c['prompt_ids']];generated=c['generated_ids'];need(len(generated)==4,'wrong output rows')
            for ids in generated:
                need(1<=len(ids)<=128 and all(type(t) is int and 0<=t<32768 for t in ids),'bad output IDs')
                need((len(ids)==128 or ids[-1]==2) and 2 not in ids[:-1],'wrong stopping')
            witness(c['after_prefill'],lengths);witness(c['after_decode'],[n+len(ids)-1 for n,ids in zip(lengths,generated)])
            need(positive(c['decode_seconds']) and len(c['decode_frames'])==max(len(ids) for ids in generated)-1,'bad decode coverage')
            active_counts=[0]*4
            for index,frame in enumerate(c['decode_frames']):
                wanted=[int(index+1<len(ids)) for ids in generated];need(frame['active']==wanted and positive(frame['gpu_ms']),'wrong inactive decode mask')
                active_counts=[a+b for a,b in zip(active_counts,wanted)]
            need(active_counts==[len(ids)-1 for ids in generated],'wrong committed decode count')
            need(len(c['prefill'])==3,'three measurements required');epoch_sets=[]
            for sample in c['prefill']:
                need(positive(sample['seconds']) and positive(sample['gpu_ms']) and positive(sample['head_gpu_ms']),'bad timing')
                epochs=sample['epochs'];need(len(epochs)==4 and len(set(epochs))==4 and all(type(e) is int and e>0 for e in epochs),'bad row leases');epoch_sets.extend(epochs)
                frames=sample['frames'];need(len(frames)==math.ceil(max(lengths)/128),'wrong prefill command count')
                for index,frame in enumerate(frames):
                    offset=index*128;remaining=min(128,max(lengths)-offset);chunk=2
                    while chunk<remaining:chunk*=2
                    active=sum(min(chunk,max(0,n-offset)) for n in lengths)
                    need(frame['offset']==offset and frame['chunk']==chunk and frame['active_tokens']==active,'wrong up-to profile/tail')
                    need(frame['packed_dispatches']==(160 if mode=='candidate' and active>=128 else 0) and positive(frame['gpu_ms']),'wrong native route')
                need(abs(sum(f['gpu_ms'] for f in frames)+sample['head_gpu_ms']-sample['gpu_ms'])<1e-6,'GPU sum mismatch')
                need(sample['predictions']==[ids[0] for ids in generated],'prefill prediction changed')
            need(len(set(epoch_sets))==12,'epochs reused between measured waves')
            current.append(dict(id=c['id'],prompt_ids=c['prompt_ids'],generated_ids=generated,after_prefill=c['after_prefill'],after_decode=c['after_decode']))
            tokens+=sum(len(x) for x in generated)
        if reference is None:reference=current
        need(current==reference,'token or complete KV/hidden parity failed')
    for cohort,(ci,ni) in enumerate([(0,1),(3,2)],1):
        for c,n in zip(records[ci]['cases'],records[ni]['cases']):
            ct=statistics.median(x['seconds'] for x in c['prefill']);nt=statistics.median(x['seconds'] for x in n['prefill']);cg=statistics.median(x['gpu_ms'] for x in c['prefill']);ng=statistics.median(x['gpu_ms'] for x in n['prefill'])
            count=sum(len(x) for x in c['generated_ids'])
            comparisons.append(dict(cohort=cohort,id=c['id'],actual_prompt_tokens=[len(p) for p in c['prompt_ids']],control_prefill_seconds=ct,candidate_prefill_seconds=nt,prefill_ratio=ct/nt,output_tokens=count,control_native_aggregate_tps=count/(c['prefill'][-1]['seconds']+c['decode_seconds']),candidate_native_aggregate_tps=count/(n['prefill'][-1]['seconds']+n['decode_seconds']),aggregate_scope='One final native ready-embedding prefill wave plus ordinary decode; state witnesses/readiness/HTTP excluded. Descriptive, not serving admission.'))
    return comparisons,tokens

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;need(not a.output.exists(),'fresh audit output required')
    fixture=json.loads((r/'fixture1.json').read_text());plan=json.loads((r/'plan1.json').read_text());records=[];inventories=[];files=[r/'fixture1.json',r/'plan1.json'];library=Path(plan['library'])/'libgarnet.dylib';need(sha(library)==plan['library_sha256'],'changed library');files.append(library)
    for stem in ['batch-cohort1-control','batch-cohort1-candidate','batch-cohort2-candidate','batch-cohort2-control']:
        status=r/(stem+'-status1.json');command=r/(stem+'-command1.json');result=r/(stem+'1/results.json');inventory=r/(stem+'1/prepared-inventory.json')
        need(json.loads(status.read_text())['returncode']==0,'nonzero process status');cmd=json.loads(command.read_text());need(cmd['library_sha256']==plan['library_sha256'] and cmd['source_sha256']==sha(Path(cmd['argv'][2])),'changed source/library');files.extend([status,command,result,inventory]);records.append(json.loads(result.read_text()));inventories.append(json.loads(inventory.read_text()))
    for path,value in fixture['sources'].items():need(sha(Path(path))==value,'changed frozen workload');files.append(Path(path))
    comparisons,tokens=validate(fixture,records,inventories);rejected=[]
    def reject(label,mutate):
        b=copy.deepcopy(records);i=copy.deepcopy(inventories);mutate(b,i)
        try:validate(fixture,b,i)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(label)
        else:raise ValueError('accepted corruption '+label)
    reject('incomplete',lambda b,i:b[0].update(completed=False))
    reject('token',lambda b,i:b[1]['cases'][0]['generated_ids'][0].__setitem__(0,(b[1]['cases'][0]['generated_ids'][0][0]+1)%32768))
    reject('kv',lambda b,i:b[2]['cases'][1]['after_decode']['state_sha256'][31].__setitem__(1,'0'*64))
    reject('hidden',lambda b,i:b[3]['cases'][1]['after_prefill'].update(hidden_sha256='1'*64))
    reject('slots',lambda b,i:b[1]['cases'][1]['after_prefill'].update(slots=[2048]*4))
    reject('route',lambda b,i:b[1]['cases'][0]['prefill'][0]['frames'][0].update(packed_dispatches=0))
    reject('tail-route',lambda b,i:b[1]['cases'][1]['prefill'][0]['frames'][-1].update(packed_dispatches=160))
    reject('epoch-reuse',lambda b,i:b[0]['cases'][0]['prefill'][1].update(epochs=b[0]['cases'][0]['prefill'][0]['epochs']))
    reject('nonfinite-time',lambda b,i:b[0]['cases'][0]['prefill'][0].update(seconds=float('nan')))
    reject('decode-mask',lambda b,i:b[1]['cases'][0]['decode_frames'][0].update(active=[0]*4))
    reject('owner-leak',lambda b,i:b[0]['cleanup'].update(garnet_live_matrix_owners=1))
    reject('scratch-size',lambda b,i:i[0]['buffers'][next(x['buffer_id'] for x in i[0]['references'] if x['role'].startswith('packed_prefill_'))].update(length_bytes=4))
    a.output.write_text(json.dumps(dict(completed=True,scope=__doc__,aggregate_gpu_time_qualified=False,aggregate_gpu_time_limitation='predict_rows does not refresh session GPU timer; head_gpu_ms duplicates prior body. Raw self-consistency checked only; wall times unaffected.',comparisons=comparisons,generated_ids_compared=tokens,request_sequences=32,exact_full_kv_and_visible_hidden_parity=True,rejected_corruptions=rejected,files={str(p.resolve()):sha(p) for p in files},no_http_or_quality_admission=True),indent=2)+'\n');print('B4 audit PASS',tokens,'IDs',len(rejected),'corruptions rejected')
if __name__=='__main__':main()
