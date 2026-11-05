#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit serial actual-weight MPS/flash late-chunk timings and output witnesses.

Warm repeated identical body work excludes embedding, head, HTTP and scheduler.
FP32 online softmax reassociates arithmetic; cross-variant bit identity is
reported, not required. Frozen complete output sequences are checked separately.
"""
import argparse,hashlib,json,math,re,statistics
from pathlib import Path

def require(value,message):
    if not value:raise ValueError(message)

def audit(root):
    root=root.resolve();m=root.parent;files={}
    def sha(p):
        with p.open('rb') as stream:files[str(p.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
        return files[str(p.resolve())]
    def read(p):sha(p);return json.loads(p.read_text())
    status=read(root/'collector-status.json')
    require(status['completed'] is True and len(status['runs'])==8,'incomplete body collection')
    identities=[(0,1,0),(0,1,1),(0,4,0),(0,4,1),(1,4,1),(1,4,0),(1,1,1),(1,1,0)]
    previous=0;results={};pids=set();fixture=m/'mistral-coding-context-fixture1/fixture.json'
    require(sha(fixture)=='6847ca35e1b1dac2bf04bb8e4f2cf358a7f7a0acb4da3fc5b3a1fff4fdbc35ad','changed frozen fixture')
    libraries={0:('mistral-compact-prefill-library2','f77445b21714e9663974a78b82f7652c41fe251e11f96de044f3d462be18ba96'),
               1:('mistral-flash-attention-library2','19bd5f3346bb3e32b5d3a6e004da1bfa0fdc6a11b84d478947cdb7db55ec3d38')}
    for identity,run in zip(identities,status['runs']):
        cohort,owners,flash=identity;label=f'cohort{cohort}-owners{owners}-flash{flash}';path=root/(label+'.json')
        require((run['cohort'],run['owners'],run['flash'])==identity and run['status']=='terminal' and run['returncode']==0 and
                run['pid']>0 and run['pid'] not in pids and previous<=run['started_ns']<run['finished_ns'],'failed/overlapping native run')
        previous=run['finished_ns'];pids.add(run['pid'])
        lib=m/libraries[flash][0]
        require(sha(lib/'libgarnet.dylib')==libraries[flash][1] and run['argv'][1:]==['-B',str(root/'tools/run_mistral_flash_stage.py'),
                str(lib),str(m/'mistral-flash-attention-source2'),str(fixture),str(path),str(owners),str(flash)],'changed runtime/command')
        data=read(path);sha(root/(label+'.log'))
        require(data['completed'] is True and data['library']==str(lib/'libgarnet.dylib') and
                (data['owners'],data['flash'],data['batch'],data['capacity'],data['chunk'],data['offset'],data['warmups'],data['repeats'])==
                (owners,flash,4,4096,128,1920,3,9),'incorrect late-chunk contract')
        tags={p.strip() for p in data['backend'].split(';')}
        require(('chunk_flash_attention_fp32_operands' in tags)==bool(flash) and 'compact_owner_prefill' in tags,'incorrect arithmetic backend')
        require(len(data['seeds'])==15 and [s['offset'] for s in data['seeds']]==list(range(0,1920,128)) and
                all(math.isfinite(s['wall_seconds']) and s['wall_seconds']>0 for s in data['seeds']),'missing real prompt seeds')
        require(len(data['measured'])==9,'missing measured body samples')
        for index,s in enumerate(data['measured']):
            require((s['repeat'],s['flash_dispatches'],s['mps_dispatches'])==(index,32*flash,32*(1-flash)) and
                    math.isfinite(s['wall_seconds']) and s['wall_seconds']>0 and math.isfinite(s['gpu_ms']) and
                    0<s['gpu_ms']<=s['wall_seconds']*1000+1,'invalid body timing/route')
        a,b,closed=data['prepared'],data['after'],data['cleanup']
        require(a['metal_allocated_bytes']==b['metal_allocated_bytes'] and
                0<=b['metal_allocated_bytes']<=b['metal_recommended_working_set_bytes']-256*1024*1024 and
                0<=closed['metal_allocated_bytes']<=4*1024*1024,'native allocation bounds differ')
        for key in ['garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets']:
            require(a[key]==b[key] and closed[key]==0,'native owner counts changed/retained')
        require(len(data['state_sha256'])==32 and all(len(layer)==2 and all(re.fullmatch('[0-9a-f]{64}',v) for v in layer)
                for layer in data['state_sha256']) and re.fullmatch('[0-9a-f]{64}',data['hidden_sha256']),'invalid complete state witness')
        require(len(data['predictions'])==4 and all((type(p) is int and 0<=p<32768) if r<owners else p==-1
                for r,p in enumerate(data['predictions'])),'prediction escaped active physical owner')
        results[identity]=data
    comparisons=[]
    for owners in (1,4):
        for flash in (0,1):
            a,b=results[0,owners,flash],results[1,owners,flash]
            require((a['state_sha256'],a['hidden_sha256'],a['predictions'])==(b['state_sha256'],b['hidden_sha256'],b['predictions']),
                    'same-variant state differs across cohorts')
        for cohort in (0,1):
            a,b=results[cohort,owners,0],results[cohort,owners,1]
            require(a['predictions']==b['predictions'],'post-prefill prediction changed between variants')
            wall_a,wall_b=[statistics.median(s['wall_seconds'] for s in item['measured']) for item in (a,b)]
            gpu_a,gpu_b=[statistics.median(s['gpu_ms'] for s in item['measured']) for item in (a,b)]
            comparisons.append(dict(cohort=cohort,owners=owners,mps_wall_seconds=wall_a,flash_wall_seconds=wall_b,
                wall_ratio=wall_a/wall_b,mps_gpu_ms=gpu_a,flash_gpu_ms=gpu_b,gpu_ratio=gpu_a/gpu_b,
                full_KV_bitwise_equal=a['state_sha256']==b['state_sha256'],hidden_bitwise_equal=a['hidden_sha256']==b['hidden_sha256']))
    for name,digest in read(root/'tools/manifest.json').items():require(sha(root/'tools'/name)==digest,'changed body driver')
    sha(Path(__file__))
    return dict(passed=True,processes=8,measured_body_forwards=72,comparisons=comparisons,files=files,scope=__doc__,
                serving_speedup_established=False,speed_targets_achieved=False,frozen_comparators_rerun=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=audit(args.collection)
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:report[k] for k in ('passed','comparisons')}))
