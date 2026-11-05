#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check actual native compact-owner cache, hidden, epoch and prefix evidence."""
import argparse,hashlib,json,math,re
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


def hashes(values):
    return len(values)==32 and all(len(layer)==2 and all(re.fullmatch('[0-9a-f]{64}',v) for v in layer) for layer in values)


def validate_parity(data):
    labels=['owner0','owner3','owners12','gapped02','all4','small-fallback','tail','owner3-large',
            'owners12-large','owners23-large','reclaimed-owner0','reset-owner3']
    require(data['completed'] is True and [r['label'] for r in data['records']]==labels,'missing/incomplete compact model cases')
    require(data['negative']==['missing-active-epoch','late-family-preparation','stale-reclaimed-epoch'],'missing native rejection cases')
    require('compact_owner_prefill' in {p.strip() for p in data['backend'].split(';')},'wrong compact backend')
    for item in data['records']:
        require(hashes(item['full_KV_sha256']) and item['full_KV_sha256']==item['reference_KV_sha256'],'full physical cache differs')
        require(re.fullmatch('[0-9a-f]{64}',item['hidden_sha256']) and item['hidden_sha256']==item['reference_hidden_sha256'],'physical hidden routing differs')
        require(item['predictions']==item['reference_predictions'] and len(item['predictions'])==4,'physical predictions differ')
        live=[r for r in range(4) if item['counts'][r]];chunk=item['chunk']
        rows=len(live) if len(live) in (1,2) and (len(live)==1 or live[1]==live[0]+1) and len(live)*chunk>4 else 0
        require(item['compact_rows']==rows and item['compact_first']==(live[0] if rows else 4),'wrong native owner-span selection')
        require(item['layout_vector_dispatches']==256+int(rows>0),'missing compact physical hidden routing')
        for row,prediction in enumerate(item['predictions']):
            require((type(prediction) is int and 0<=prediction<32768) if row in live else prediction==-1,'prediction escaped active physical owner')
        require(math.isfinite(item['diagnostic_wall_seconds']) and item['diagnostic_wall_seconds']>0,'invalid diagnostic time')
    return dict(cases=len(labels),full_physical_KV_hidden_and_prediction_parity=True)


def validate_prefix(data,fixture,controls):
    labels=['A-initial','E-replacement','A-exact-return','append-fresh','append-restored','changed-fresh','changed-restored']
    require(data['completed'] is True and [r['label'] for r in data['records']]==labels,'missing/incomplete prefix cases')
    require(data['prepared']['compact_families']==14 and data['prepared']['mps_plans']==21
            and data['prepared']['mps_scratch']==1044162560 and data['prepared']['mps_buckets']==126,'incorrect prepared compact memory/profile inventory')
    by_label={r['label']:r for r in data['records']};cases=[c for c in fixture['cases'] if c['kind']=='coding']
    for label,index,budget in [('A-initial',0,32),('E-replacement',4,16),('A-exact-return',0,32)]:
        control=controls[cases[index]['id']];item=by_label[label]
        require(item['prompt_ids']==cases[index]['prompt_ids']==control['prompt_ids']
                and item['generated_ids']==control['generated_ids'][:budget] and item['exact_control'] is True,'frozen prefix token control changed')
    for fresh,restored in [('append-fresh','append-restored'),('changed-fresh','changed-restored')]:
        a,b=by_label[fresh],by_label[restored]
        require(a['prompt_ids']==b['prompt_ids'] and a['generated_ids']==b['generated_ids'] and b['exact_control'] is True,'cached/fresh continuation differs')
    require(len(data['rejections'])==3 and set(data['rejections'])=={'stale-restore','stale-capture','stale-nonzero-restore'},'missing stale prefix epoch rejection')
    for frame in data['frames']:
        if frame['stage']!='prefill':continue
        live=[r for r in range(4) if frame['valid'][r]];rows=len(live)
        expected=rows if rows in (1,2) and (rows==1 or live[1]==live[0]+1) and rows*frame['chunk']>4 else 0
        require(frame['compact_rows']==expected and frame['compact_first']==(live[0] if expected else 4),'prefix body used wrong physical span')
    nonzero=data['nonzero'];require(nonzero['row']==3 and nonzero['history_tokens']==nonzero['restored_tokens']==len(nonzero['history_ids'])
            and hashes(nonzero['before_state']) and nonzero['before_state']==nonzero['after_state']
            and 0<=nonzero['before_prediction']<32768 and nonzero['before_prediction']==nonzero['after_prediction'],
            'nonzero compact prefix restore changed full state/prediction')
    require(nonzero['history_ids']==cases[3]['prompt_ids']+controls[cases[3]['id']]['generated_ids'][:176]+cases[0]['prompt_ids'][:7],
            'nonzero prefix token history differs from frozen survivor continuation')
    for row in (1,2,3):
        actual=(next(s['generated_ids'] for s in data['survivor_rows'] if s['row']==row) if row!=3 else nonzero['survivor_ids'])
        require(len(actual)==176 and actual==controls[cases[row]['id']]['generated_ids'][:176],'running survivor changed')
    return dict(cases=7,frozen_control_comparisons=3,fresh_cached_comparisons=2,survivor_rows=3,survivor_tokens=528,
                nonzero_prefix_tokens=nonzero['restored_tokens'],nonzero_complete_cache_and_prediction_parity=True)


def audit(parity,prefix):
    files={}
    def read(path):
        raw=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
    def bind(path):
        with path.open('rb') as stream:files[str(path.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
    m=parity.parent;fixture=read(m/'mistral-coding-context-fixture1/fixture.json')
    require(files[str((m/'mistral-coding-context-fixture1/fixture.json').resolve())]=='6847ca35e1b1dac2bf04bb8e4f2cf358a7f7a0acb4da3fc5b3a1fff4fdbc35ad','changed frozen fixture')
    controls={}
    for case in [c for c in fixture['cases'] if c['kind']=='coding'][:5]:controls[case['id']]=read(m/'mistral-native-4k-coding1'/case['id']/'request-0.result.json')
    data=read(parity/'run/status.json');pdata=read(prefix/'result.json');results=dict(parity=validate_parity(data),prefix=validate_prefix(pdata,fixture,controls))
    previous=0
    for root,status in [(parity,data),(prefix,pdata)]:
        collector=read(root/'collector-status.json')
        require(collector['status']=='terminal' and collector['returncode']==0 and collector['pid']>0
                and previous<=collector['started_ns']<collector['finished_ns'],'failed/overlapping model processes')
        previous=collector['finished_ns']
        require(status['library']==str(m/'mistral-compact-prefill-library2/libgarnet.dylib'),'wrong native library binding')
        for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
            require(status['cleanup'][key]==0,'native owners retained')
        require(0<=status['cleanup']['metal_allocated_bytes']<=4*1024*1024,'native close allocation bound exceeded')
        for memory in [status['prepared']]+[r['memory'] for r in status.get('records',[]) if 'memory' in r]:
            require(0<=memory['metal_allocated_bytes']<=memory['metal_recommended_working_set_bytes']-256*1024*1024,
                    'native sampled allocation exceeds compiled working set')
        manifest=read(root/'tools/manifest.json')
        for name,digest in manifest.items():bind(root/'tools'/name);require(files[str((root/'tools'/name).resolve())]==digest,'changed model driver')
        bind(root/'collector.log')
    bind(m/'mistral-compact-prefill-library2/libgarnet.dylib')
    require(files[str((m/'mistral-compact-prefill-library2/libgarnet.dylib').resolve())]=='f77445b21714e9663974a78b82f7652c41fe251e11f96de044f3d462be18ba96','changed native candidate')
    bind(Path(__file__).resolve())
    return dict(passed=True,results=results,files=files,scope=__doc__,speed_established=False,new_coding_quality_score_established=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parity','prefix','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();r=audit(a.parity.resolve(),a.prefix.resolve())
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps({'passed':r['passed'],'results':r['results']}))
