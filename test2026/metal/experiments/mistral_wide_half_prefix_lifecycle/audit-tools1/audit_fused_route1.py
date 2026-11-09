from audit_half_transport1 import require
def fused_check(events,profile):
    require(profile['fused_mlp_prefill'] is True,'missing fused policy')
    ready=[e for e in events if e['event']=='ready'];require(len(ready)==1 and 'fused_mistral_mlp_prefill_fast_v1' in ready[0]['backend'],'wrong fused backend')
    prep=[e for e in events if e['event']=='native_fused_mlp_prepared']
    require(prep==[dict(event='native_fused_mlp_prepared',sites=[1]*32,scratch_bytes=14680064,producer='fused-mistral-mlp-fast-v1')] and events.index(prep[0])<events.index(ready[0]),'wrong fused preparation')
    batches=[e for e in events if e['event']=='native_batch' and (e['stage']!='prefill' or any(q['prefill_tokens'] for q in e['requests']))];fused=[e for e in events if e['event']=='native_fused_mlp'];spans=[e for e in events if e['event']=='native_compact_prefill']
    require(len(batches)==len(fused)==len(spans)>0,'missing fused/compact observations')
    maximum_parent=maximum_gpu=0
    for batch,event,span in zip(batches,fused,spans):
        prefill=batch['stage']=='prefill';chunk=batch['prefill_chunk'] if prefill else 1
        parent=4*chunk;active=sum(r['prefill_tokens'] if prefill else 1 for r in batch['requests'])
        rows=sorted(r['row'] for r in batch['requests'] if not prefill or r['prefill_tokens'])
        compact=prefill and len(rows) in [1,2] and len(rows)*chunk>4 and (len(rows)==1 or rows[1]==rows[0]+1)
        require(span==dict(event='native_compact_prefill',stage=batch['stage'],rows=len(rows) if compact else 0,first=rows[0] if compact else 4,chunk=chunk),'wrong compact owner mapping')
        gpu=(len(rows) if compact else 4)*chunk
        require(0<active<=gpu<=512 and parent>=gpu,'effective GPU projection exceeds compiled token workspace')
        blocks=32 if prefill and active>=128 and gpu>4 else 0
        # This worker field reports the full B4 input parent, before compaction.
        require(event==dict(event='native_fused_mlp',blocks=blocks,active_tokens=active,physical_rows=parent,logical_affine_products=blocks*3,producer_kernels=blocks,down_kernels=blocks),'wrong fused route/parent metadata')
        maximum_parent=max(maximum_parent,parent);maximum_gpu=max(maximum_gpu,gpu)
    require(sum(e['blocks'] for e in fused)>0,'fusion never executed')
    return dict(blocks=sum(e['blocks'] for e in fused),producer_kernels=sum(e['producer_kernels'] for e in fused),down_kernels=sum(e['down_kernels'] for e in fused),maximum_declared_input_parent_rows=maximum_parent,maximum_effective_gpu_rows=maximum_gpu,parent_metadata_is_not_gpu_dispatch_extent=True)
