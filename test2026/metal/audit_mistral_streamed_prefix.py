#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit B4 prefix/scheduler state evidence, retaining the cold-FP16 limitation."""
import argparse,copy,hashlib,json
from pathlib import Path

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def hashword(h):return type(h) is str and len(h)==64 and all(c in '0123456789abcdef' for c in h)
def state(x):need(len(x)==32 and all(len(p)==2 and all(hashword(h) for h in p) for p in x),'KV hash geometry')
def ids(x,count):need(len(x)==count and all(type(t) is int and 0<=t<32768 for t in x),'invalid fixed-budget IDs')
def signature(x,length,hidden):
    state(x['kv']);need(x['position']==length and type(x['prediction']) is int and 0<=x['prediction']<32768,'invalid state boundary')
    need(hashword(x['hidden_sha256']) if hidden else x['hidden_sha256'] is None,'incorrect visible hidden scope')
def validate(fixture,reports):
    need(len(reports)==2,'control/candidate required');canonical=[]
    prompts=[p[:512] for p in fixture['cases'][1]['prompts']];a=prompts[0];e=list(a);e[0]=17;changed=list(a);changed[300]=(changed[300]+17)%32768
    labels=['A-initial','E-replacement','A-exact','A-appended','A-changed','A-shorter']
    for d,mode in zip(reports,['control','candidate']):
        need(d['completed'] is True and d['mode']==mode,'incomplete collection')
        need(d['ready']['graphs']==224 and d['ready']['plans']==3 and d['host_cache_bytes']==384*1024*1024 and d['cache_entries']==16,'changed prepared profile')
        need('mistral_streamed_prefill_optional' in d['backend'],'wrong backend')
        episodes=d['episodes'];need([x['label'] for x in episodes]==labels,'missing episode')
        inputs=[a,e,a,a+episodes[0]['generated_ids'],changed,a[:256]]
        for index,(x,prompt) in enumerate(zip(episodes,inputs)):
            need(x['prompt_ids']==prompt,'changed prefix query');ids(x['generated_ids'],8);signature(x['prefill'],len(prompt),False);signature(x['decode'],len(prompt)+7,True)
            need(x['prefill']['prediction']==x['generated_ids'][0],'wrong initial prediction')
            need(x['survivor_before']==[1+index*8]*3 and x['survivor_after']==[1+(index+1)*8]*3,'survivor progress changed')
            need(x['cache']['accounted_host_bytes']<=d['host_cache_bytes'],'host cache exceeded')
        need(episodes[0]['generated_ids']==episodes[2]['generated_ids'] and episodes[0]['prefill']==episodes[2]['prefill'] and episodes[0]['decode']==episodes[2]['decode'],'exact return differs')
        need([c['tokens'] for c in d['captures']]==[64,128,512],'changed checkpoint boundaries')
        for i,c in enumerate(d['captures']):
            st=c['cache'];need(st['entries']==i+1 and st['hits']==st['misses']==0 and sum(v['tokens'] for v in d['captures'][:i+1])*262144<=st['accounted_host_bytes']<=d['host_cache_bytes'],'snapshot accounting')
        direct=episodes[-1]['cache'];need(direct['hits']==4 and direct['misses']==1 and direct['entries']==3 and direct['evictions']==0,'direct cache semantics changed')
        need(len(d['restores'])==5,'missing restore cases');stored=[a[:64],a[:128],a]
        for r,q in zip(d['restores'],[e,a,inputs[3],changed,a[:256]]):
            want=max([len(p) for p in stored if len(p)<=len(q) and q[:len(p)]==p]+[0]);need(r['prompt_ids']==q and r['retained']==want,'not longest stored prefix')
        need(len(d['stability'])==14,'missing survivor stability checks')
        for item in d['stability']:
            need(item['before']==item['after'] and len(item['before'])==3,'inactive survivor changed')
            for x in item['before']:state(x)
        epoch_restore={r['epoch']:r['retained'] for r in d['restores']};last_epochs=[0]*4;positions=[0]*4;streamed=0
        for f in d['frames']:
            epochs=f['epochs'];need(len(epochs)==4 and len(f['before'])==len(f['after'])==4,'bad frame width')
            for row in range(4):
                if epochs[row]!=last_epochs[row]:positions[row]=epoch_restore.get(epochs[row],0) if row==0 else 0
            need(f['before']==positions,'incorrect restored/fresh starting position')
            if f['stage']=='prefill':
                counts=f['counts'];need(f['chunk'] in [2,4,8,16,32,64,128] and len(counts)==4 and all(type(c) is int and 0<=c<=f['chunk'] for c in counts),'invalid up-to chunk')
                wanted=160 if mode=='candidate' and sum(counts)==512 else 0
            else:
                need(f['stage']=='decode','unknown stage');counts=f['active'];need(len(counts)==4 and all(c in [0,1] for c in counts),'bad decode mask');wanted=0
            need(all(not counts[row] or epochs[row]>0 for row in range(4)) and f['streamed_dispatches']==wanted,'wrong native route/lease')
            positions=[p+c for p,c in zip(positions,counts)];need(f['after']==positions and max(positions)<=4096,'incorrect progress');last_epochs=list(epochs);streamed+=wanted
        need(streamed==(640 if mode=='candidate' else 0),'missing full B4 streamed execution')
        need(len(d['survivors'])==3 and [x['row'] for x in d['survivors']]==[1,2,3],'missing survivor references')
        for row,x in enumerate(d['survivors'],1):need(x['prompt_ids']==prompts[row],'survivor prompt changed');ids(x['generated_ids'],49);signature(x['signature'],560,True)
        targets=[episodes[0],episodes[1]]+episodes[3:]+d['survivors'];need(len(d['fresh'])==len(targets)==8,'missing lookup-free controls')
        for fresh,target in zip(d['fresh'],targets):
            want=target.get('decode',target.get('signature'));need(fresh['generated_ids']==target['generated_ids'] and fresh['decode']==want,'fresh state/token divergence')
            if 'prefill' in target:need(fresh['prefill']==target['prefill'],'fresh prefill differs')
        need(d['rejected']==['stale-restore','stale-capture','uninitialized-prediction','late-profile-preparation','scheduler-stale-dispatch'],'missing guard cases')
        leases=d['leases'];need(len({x['epoch'] for x in leases})==len(leases) and all(x['epoch']>0 and 0<=x['row']<4 for x in leases),'lease epoch reused')
        s=d['scheduler'];done=s['records'];need(set(done)=={'A','B','C','D','E','A-return','A-append'} and s['mixed_ticks']==3 and s['head_only']==1,'missing continuous transitions')
        need(len({x['request_id'] for x in done.values()})==7,'scheduler IDs reused')
        for label,x in done.items():ids(x['generated_ids'],32 if label in ['B','C','D'] else 2);state(x['state_sha256']);need(hashword(x['hidden_sha256']),'missing decode hidden')
        need(all(done[k]['row']==0 for k in ['A','E','A-return','A-append']) and len({done[k]['epoch'] for k in ['A','E','A-return','A-append']})==4,'A/E placement/epoch changed')
        need(done['A']['generated_ids']==done['A-return']['generated_ids'] and done['A']['state_sha256']==done['A-return']['state_sha256'] and done['A']['hidden_sha256']==done['A-return']['hidden_sha256'],'scheduler exact return differs')
        streamed_policy=0;head=0
        for p in s['plans']:
            requests=p['requests'];count=sum(r['prefill_tokens'] if p['stage']=='prefill' else 1 for r in requests);need(p['active_tokens']==count,'wrong plan token count')
            wanted=160 if mode=='candidate' and p['stage']=='prefill' and count==512 else 0;need(p['streamed_dispatches']==wanted,'wrong scheduler native route');streamed_policy+=wanted
            if p['head_only']:
                need(p['stage']=='prefill' and len(requests)==1 and requests[0]['id']==done['A-return']['request_id'] and requests[0]['cached_tokens']==128 and requests[0]['prefill_tokens']==0,'invalid exact head-only hit');head+=1
            for req in requests:
                if p['stage']=='prefill' and req['id']==done['A-append']['request_id']:need(req['cached_tokens']==129 and req['prefill_offset']==129 and req['prefill_tokens']==1,'decoded history boundary changed')
        need(streamed_policy==(160 if mode=='candidate' else 0) and head==1,'missing scheduler streaming/head coverage')
        need(s['cache']['accounted_host_bytes']<=d['host_cache_bytes'] and s['cache']['hits']==6 and s['cache']['misses']==6 and s['cache']['evictions']>=1,'incorrect bounded cache/lookup accounting')
        need(s['status']['active']==s['status']['tracked']==0,'scheduler not drained')
        need([x['label'] for x in d['policy_fresh']]==['A','B','C','D','E','A-append'],'missing scheduler fresh oracles')
        for f in d['policy_fresh']:
            target=done[f['label']];need(f['generated_ids']==target['generated_ids'] and f['signature']['kv']==target['state_sha256'] and f['signature']['hidden_sha256']==target['hidden_sha256'],'scheduler matching-lineage parity failed')
            signature(f['signature'],len(target['prompt_ids'])+len(target['generated_ids'])-1,True)
            if f['label']=='A-append':
                need(f['precision_recipe']==dict(prefill_tokens=128,fp32_decode_tokens=1,remaining_prefill_tokens=1,remaining_prefill_chunk=128),'incorrect decode lineage oracle')
                cold=f['cold_fp16'];need(cold is not None and cold['generated_ids']==target['generated_ids'] and cold['state_equal'] is False and cold['signature']['kv']!=target['state_sha256'],'cold FP16 limitation hidden')
            else:need(f['cold_fp16'] is None and f['precision_recipe']['fp32_decode_tokens']==0,'unexpected oracle policy')
        for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:need(d['cleanup'][key]==0,'native owner leak')
        need(d['cleanup']['metal_allocated_bytes']<=4*1024*1024,'device retained')
        # No precision/timing differences are normalized out of these witnesses.
        canonical.append(dict(episodes=d['episodes'],survivors=d['survivors'],fresh=d['fresh'],policy_fresh=d['policy_fresh'],scheduler_records=done,stability=d['stability'],captures=d['captures'],restores=d['restores']))
    need(canonical[0]==canonical[1],'control/candidate state or request ownership differs')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;need(not a.output.exists(),'fresh output required')
    fixture=json.loads((r/'fixture1.json').read_text());reports=[];files=[r/'fixture1.json'];plan=json.loads((r/'plan1.json').read_text());library=Path(plan['library'])/'libgarnet.dylib';need(sha(library)==plan['library_sha256'],'library changed');files.append(library)
    for mode in ['control','candidate']:
        stem='prefix-'+mode;status=r/(stem+'-status2.json');command=r/(stem+'-command2.json');result=r/(stem+'2/results.json');need(json.loads(status.read_text())['returncode']==0,'failed collection');cmd=json.loads(command.read_text());need(cmd['library_sha256']==sha(library) and cmd['source_sha256']==sha(Path(cmd['argv'][2])),'source/library changed');reports.append(json.loads(result.read_text()));files.extend([status,command,result,Path(cmd['argv'][2])])
    validate(fixture,reports);rejections=[]
    def reject(label,mutate):
        b=copy.deepcopy(reports);mutate(b)
        try:validate(fixture,b)
        except (ValueError,KeyError,IndexError,TypeError):rejections.append(label)
        else:raise ValueError('accepted corruption '+label)
    reject('incomplete',lambda b:b[0].update(completed=False))
    reject('wrong-retained-prefix',lambda b:b[1]['restores'][3].update(retained=300))
    reject('survivor-kv',lambda b:b[1]['stability'][0]['after'][0][0].__setitem__(0,'0'*64))
    reject('fresh-token',lambda b:b[0]['fresh'][0]['generated_ids'].__setitem__(0,(b[0]['fresh'][0]['generated_ids'][0]+1)%32768))
    reject('active-route',lambda b:next(f for f in b[1]['frames'] if f['streamed_dispatches']).update(streamed_dispatches=0))
    reject('restored-position',lambda b:b[0]['frames'][0].update(before=[1,0,0,0]))
    reject('missing-head-hit',lambda b:b[1]['scheduler'].update(head_only=0))
    reject('stale-epoch',lambda b:b[1]['scheduler']['records']['A-return'].update(epoch=b[1]['scheduler']['records']['A']['epoch']))
    reject('lineage-erased',lambda b:b[0]['policy_fresh'][-1]['precision_recipe'].update(fp32_decode_tokens=0))
    reject('cold-limitation-erased',lambda b:b[1]['policy_fresh'][-1]['cold_fp16'].update(state_equal=True))
    reject('host-budget',lambda b:b[0]['scheduler']['cache'].update(accounted_host_bytes=512*1024*1024))
    reject('owner-leak',lambda b:b[0]['cleanup'].update(garnet_live_matrix_owners=1))
    a.output.write_text(json.dumps(dict(completed=True,scope=__doc__,matching_precision_lineage_passed=True,cold_fp16_state_equivalence=False,streamed_commands_per_candidate=5,lookup_free_references_per_process=14,rejected_corruptions=rejections,files={str(p.resolve()):sha(p) for p in set(files)},no_broad_cached_http_or_coding_admission=True),indent=2)+'\n');print('prefix audit PASS',len(rejections),'corruptions; cold FP16 equivalence remains unadmitted')
if __name__=='__main__':main()
