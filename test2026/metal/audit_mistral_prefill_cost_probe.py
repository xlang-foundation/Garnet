#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize serialized actual-weight operation diagnosis; not serving latency."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path


def require(value,message):
    if not value:raise ValueError(message)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.collection.resolve();files={}
    def read(path):
        data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return json.loads(data)
    collector=read(root/'collector-status.json');status=read(root/'probe.json')
    require(collector['returncode']==0 and type(collector['pid']) is int and collector['pid']>0
            and collector['started_ns']<collector['finished_ns'],'nonterminal/failed cost probe')
    require(status['completed'] is True and status['case_id']=='context-2048' and status['batch']==4
            and status['chunk']==128 and status['profile_offset']==1920 and len(status['records'])==16,'wrong diagnostic profile')
    require(Path(collector['argv'][7])==root/'probe.json' and collector['argv'][6]=='context-2048'
            and collector['argv'][8:]==['4','128','1920'],'cost probe command/profile differs')
    library=Path(collector['argv'][3])/'libgarnet.dylib'
    with library.open('rb') as f:files[str(library)]=hashlib.file_digest(f,'sha256').hexdigest()
    require(status['library']==str(library) and files[str(library)]==collector['library_sha256'],'cost probe library changed/differs')
    fixture=read(Path(collector['argv'][5]));require(len(next(c for c in fixture['cases'] if c['id']=='context-2048')['prompt_ids'])==2056,'diagnostic prompt changed')
    for i,record in enumerate(status['records']):
        require(record['offset']==i*128 and record['valid_tokens']==128 and record['profiled']==(i==15)
                and math.isfinite(record['wall_seconds']) and record['wall_seconds']>0,'incomplete/changed chunk sequence')
        m=record['memory'];require(0<=m['metal_allocated_bytes']<=m['metal_recommended_working_set_bytes']-256*1024*1024,'sampled working set exceeded guard')
    for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):require(status['cleanup'][key]==0,'retained native owners')
    require(0<=status['cleanup']['metal_allocated_bytes']<=4*1024*1024,'close allocation guard exceeded')
    log=root/'probe.log';data=log.read_bytes();files[str(log)]=hashlib.sha256(data).hexdigest()
    ops=[json.loads(line.split('metal_op_profile ',1)[1]) for line in data.decode().splitlines() if line.startswith('metal_op_profile ')]
    kinds=['Layout','Layout','Norm','Layout','Layout','Layout','MatMul','MatMul','MatMul','NormRope','NormRope','AttentionChunk','KVChunk','KVChunk','MatMul','Layout','Layout','Add','Layout','Norm','MatMul','MatMul','Silu','MatMul','Add','Layout']
    require(len(ops)==832 and [o['kind'] for o in ops]==kinds*32,'missing/reordered layer operations')
    for op in ops:require(op['serialized_diagnostic'] is True and math.isfinite(op['gpu_ms']) and op['gpu_ms']>0,'invalid operation duration')
    groups=collections.defaultdict(list)
    for op in ops:groups[op['kind']].append(op['gpu_ms'])
    manifest=read(root/'tools/manifest.json')
    for name,digest in manifest.items():
        path=root/'tools'/name;data=path.read_bytes();files[str(path)]=hashlib.sha256(data).hexdigest()
        require(files[str(path)]==digest,'captured probe tool changed')
    for path in (Path(collector['argv'][2]),Path(__file__).resolve()):files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(passed=True,operations=832,serialized_gpu_duration_sum_ms=sum(o['gpu_ms'] for o in ops),
        by_kind={k:dict(count=len(v),gpu_ms_sum=sum(v),gpu_ms_median=statistics.median(v)) for k,v in groups.items()},
        unprofiled_chunk_wall_seconds=[r['wall_seconds'] for r in status['records'][:-1]],
        files=files,scope=__doc__,coding_quality_established=False,serving_speed_established=False,
        note='Only row0 has live prompt tokens. Other physical rows remain in the B4 graph. Individual operations split submissions; summed timings and fractions are diagnostic, not production attribution or token rates.')
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('serialized_operation_diagnosis_passed',True)
