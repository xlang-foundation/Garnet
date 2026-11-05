#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit paired same-session compact-owner body latency and complete state.

Repeated identical late-chunk work excludes embedding, head, streaming and
request scheduling. This audit does not establish serving throughput.
"""
import argparse,copy,hashlib,json,math,re,statistics
from pathlib import Path

OWNERS=('garnet_live_matrix_owners','garnet_live_affine_weight_owners',
        'garnet_live_bf16_weight_owners','garnet_live_pipeline_sets')

def require(value,message):
    if not value:raise ValueError(message)

def bounded(memory):
    require(0<=memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-256*1024*1024,
            'sampled allocation exceeds working set')

def validate(data,reverse):
    require(data['completed'] is True and data['reverse']==reverse and
            (data['case_id'],data['batch'],data['capacity'],data['chunk'],data['offset'],data['warmups'],data['repeats'])
            ==('context-2048',4,4096,128,1920,3,9),'incorrect paired experiment contract')
    require(data['inventory']==[14,21,1044162560,126],'incorrect prepared family inventory')
    tags={s.strip() for s in data['backend'].split(';')}
    require({'compact_owner_prefill','chunk_layout_vector4','chunk_kv_scatter_contiguous',
             'affine_coefficient_dtype=fp16'}<=tags,'incorrect native backend')
    bounded(data['prepared'])
    require([(g['label'],g['live']) for g in data['groups']]==
            [('owner0',[0]),('owner3',[3]),('owners12',[1,2])],'missing physical owner cases')
    order=['control','compact'] if reverse==0 else ['compact','control'];comparisons=[]
    for group in data['groups']:
        require([s['offset'] for s in group['seeds']]==list(range(0,1920,128)) and
                all(math.isfinite(s['wall_seconds']) and s['wall_seconds']>0 for s in group['seeds']),
                'invalid prompt seed coverage')
        require([v['variant'] for v in group['variants']]==order,'incorrect paired execution order')
        by_variant={v['variant']:v for v in group['variants']}
        for variant,item in by_variant.items():
            samples=item['measured'];rows=len(group['live']) if variant=='compact' else 0
            require(len(samples)==9,'missing measured forwards')
            for index,sample in enumerate(samples):
                require((sample['repeat'],sample['compact_rows'],sample['first'],sample['layout_dispatches'])==
                        (index,rows,group['live'][0] if rows else 4,256+int(bool(rows))),
                        'wrong native compact owner route')
                require(math.isfinite(sample['wall_seconds']) and sample['wall_seconds']>0 and
                        math.isfinite(sample['gpu_ms']) and 0<sample['gpu_ms']<=sample['wall_seconds']*1000+1,
                        'invalid measured duration')
            state=item['state_sha256']
            require(len(state['KV'])==32 and all(len(layer)==2 and all(re.fullmatch('[0-9a-f]{64}',h) for h in layer)
                    for layer in state['KV']) and re.fullmatch('[0-9a-f]{64}',state['hidden']), 'invalid full-state witness')
            a,b=item['steady'],item['after'];bounded(a);bounded(b)
            require(a['metal_allocated_bytes']==b['metal_allocated_bytes'] and all(a[k]==b[k] for k in OWNERS),
                    'native steady allocation/owners changed')
        a,b=[by_variant[v] for v in ('control','compact')]
        require(a['state_sha256']==b['state_sha256'],'full physical cache/hidden differs')
        wall_a,wall_b=[statistics.median(s['wall_seconds'] for s in item['measured']) for item in (a,b)]
        gpu_a,gpu_b=[statistics.median(s['gpu_ms'] for s in item['measured']) for item in (a,b)]
        comparisons.append(dict(case=group['label'],cohort=reverse+1,control_wall_seconds=wall_a,
            compact_wall_seconds=wall_b,wall_ratio=wall_a/wall_b,control_gpu_ms=gpu_a,compact_gpu_ms=gpu_b,gpu_ratio=gpu_a/gpu_b))
    require(all(data['cleanup'][k]==0 for k in OWNERS) and 0<=data['cleanup']['metal_allocated_bytes']<=4*1024*1024,
            'native owners/allocation retained')
    return comparisons

def audit(root):
    root=root.resolve();m=root.parent;files={};previous=0;results=[];datasets=[]
    def sha(path):
        with path.open('rb') as stream:files[str(path.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
        return files[str(path.resolve())]
    def read(path):sha(path);return json.loads(path.read_text())
    fixture=m/'mistral-coding-context-fixture1/fixture.json'
    require(sha(fixture)=='6847ca35e1b1dac2bf04bb8e4f2cf358a7f7a0acb4da3fc5b3a1fff4fdbc35ad','changed frozen fixture')
    lib=m/'mistral-compact-prefill-library2';source=m/'mistral-compact-prefill-source2'
    require(sha(lib/'libgarnet.dylib')=='f77445b21714e9663974a78b82f7652c41fe251e11f96de044f3d462be18ba96','changed native library')
    manifest=read(root/'driver-manifest.json')
    driver=root/'tools/run_mistral_compact_prefill_stage.py'
    require(manifest=={str(driver):sha(driver)},'changed frozen driver')
    for index in (1,2):
        status=read(root/f'cohort-{index}-status.json');path=root/f'cohort-{index}.json'
        require(status['status']=='terminal' and status['returncode']==0 and status['pid']>0 and
                previous<=status['started_ns']<status['finished_ns'],'failed/overlapping native process')
        previous=status['finished_ns']
        require(status['argv'][1:]==['-B',str(driver),str(lib),str(source),str(fixture),str(path),str(index-1)],'changed native command')
        data=read(path);sha(root/f'cohort-{index}.log')
        require(data['library']==str(lib/'libgarnet.dylib'),'incorrect runtime library binding')
        results.extend(validate(data,index-1));datasets.append(data)
    for a,b in zip(datasets[0]['groups'],datasets[1]['groups']):
        require(a['variants'][0]['state_sha256']==b['variants'][0]['state_sha256'],'full physical state differs across cohorts')
    # These mutations check the offline gate; they do not replace native tests.
    rejected=[]
    for label in ['missing-case','wrong-inventory','wrong-span','wrong-cache','wrong-hidden','missing-sample','retained-owner']:
        data=copy.deepcopy(datasets[0]);item=data['groups'][0]['variants'][1]
        if label=='missing-case':data['groups'].pop()
        elif label=='wrong-inventory':data['inventory'][0]-=1
        elif label=='wrong-span':item['measured'][0]['first']=3
        elif label=='wrong-cache':item['state_sha256']['KV'][0][0]='0'*64
        elif label=='wrong-hidden':item['state_sha256']['hidden']='0'*64
        elif label=='missing-sample':item['measured'].pop()
        else:data['cleanup'][OWNERS[0]]=1
        try:validate(data,0)
        except ValueError:rejected.append(label)
        else:raise AssertionError('Accepted corruption: '+label)
    sha(Path(__file__))
    return dict(passed=True,processes=2,measured_chunk_forwards=108,comparisons=results,
                complete_state_exact_across_variants_and_cohorts=True,corruptions_rejected=rejected,
                files=files,scope=__doc__,serving_speed_established=False,frozen_comparators_rerun=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=audit(args.collection)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','comparisons')}))
