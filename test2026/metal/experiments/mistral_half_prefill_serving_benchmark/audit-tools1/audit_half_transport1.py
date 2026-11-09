#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit uncached Mistral B4 HTTP admission against frozen inputs and controls.

Checks saved native scheduling/dispatch traces and end-to-end SSE token identity.
These logs do not independently observe GPU KV contents, dynamic arrivals,
peak memory, coding quality, or repeated performance.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path


OWNERS = ('garnet_live_matrix_owners', 'garnet_live_affine_weight_owners',
          'garnet_live_bf16_weight_owners', 'garnet_live_pipeline_sets')


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(events, clients, controls, profile, exit_status, expected_requests=22, control_prefixes=False,
             maximum_chunk=128,prefix_policy=False,required_active_rows=4):
    require(exit_status == dict(returncode=0, expected_close=True), 'unclean worker exit')
    counts = collections.Counter(e['event'] for e in events)
    require(counts['ready'] == counts['closed'] == counts['native_dense_schedule_bound'] == 1,
            'missing/duplicate lifecycle event')
    require(events[-1]['event'] == 'closed', 'events after close')
    require(not counts['error'], 'native error')
    bound = next(e for e in events if e['event'] == 'native_dense_schedule_bound')
    wide_owner = profile.get('wide_owner_prefill', False)
    require(type(wide_owner) is bool, 'invalid owner profile policy')
    require((bound['physical_rows'], bound['global_token_budget'], bound['maximum_chunk']) == (4, 512, 512 if wide_owner else 128)
            and bound['oversized_chunk_rejected'] is True and bound['whole_prompt_rejected'] is True,
            'incorrect compiled scheduler bound')
    ready = next(e for e in events if e['event'] == 'ready')
    require(ready['profile'] == profile and profile['model_family'] == 'mistral_7b'
            and profile['capacity'] == 4096 and profile['batch'] == 4
            and profile['prefill_chunk'] == maximum_chunk and profile['prefix_cache'] is prefix_policy
            and ready['prefix_cache'] is prefix_policy and ready['multimodal'] is False,
            'incorrect serving profile')
    backend = {p.strip() for p in ready['backend'].split(';')}
    require({'prepared_dense_prefill', 'dense_prefill_max_tokens=512',
             'indexed_token_projections', 'active_decode_projections',
             'affine_coefficient_dtype=fp16'} <= backend, 'incorrect native backend')
    grouped = profile.get('mistral_grouped_value_context',False)
    require(type(grouped) is bool and grouped==('mistral_grouped_value_context_two_heads' in backend), 'grouped value context compiled policy differs')
    aliases = profile.get('prefill_alias_groups',False)
    hidden_pool = profile.get('prefill_hidden_pool',False)
    require(type(aliases) is bool and aliases==('prefill_alias_groups' in backend), 'alias compiled policy differs')
    require(type(hidden_pool) is bool and hidden_pool==('native_prefill_hidden_pool' in backend), 'hidden pool compiled policy differs')
    if aliases:
        require(profile.get('require_lifetime_reuse') is True and 'prefill_lifetime_reuse' in backend, 'alias lifetime profile differs')
        require(counts['native_prefill_alias_prepared']==1, 'missing alias preparation')
        alias_event=next(e for e in events if e['event']=='native_prefill_alias_prepared')
        require(alias_event['layout']==profile.get('expected_prefill_alias_layout') and len(alias_event['layout'])==32, 'alias prepared layout differs')
        require(sum(x['groups'] for x in alias_event['layout'])>0 and all(x['slots']>=2*x['groups'] for x in alias_event['layout']), 'empty or invalid alias layout')
        require(events.index(alias_event)<events.index(ready), 'alias preparation after readiness')
    else: require(counts['native_prefill_alias_prepared']==0, 'unexpected alias preparation')
    half_attention = profile.get('half_attention_policy', False)
    require(half_attention is True and 'fp16_prefill_all_p16_gpupack_v1' in backend, 'wrong native half producer')
    require([e for e in events if e['event']=='native_chunk_half_attention_prepared']==[dict(event='native_chunk_half_attention_prepared',enabled=True,scratch_bytes=71303552)], 'wrong half preparation')
    matrix_attention = profile.get('chunk_attention_mps', False)
    attention_scratch = profile.get('chunk_attention_scratch', False)
    require(type(attention_scratch) is bool and attention_scratch == ('chunk_attention_full_write_scratch' in backend)
            and (not attention_scratch or matrix_attention), 'attention scratch policy differs from model/backend')
    flash_attention = profile.get('chunk_flash_attention', False)
    require(type(flash_attention) is bool and flash_attention == ('chunk_flash_attention_fp32_operands' in backend)
            and (not flash_attention or matrix_attention), 'flash attention policy differs from model/backend')
    active_requests = profile.get('chunk_attention_active_rows', False)
    mistral_reuse = profile.get('mistral_decode_reuse', False)
    kv_scatter = profile.get('chunk_kv_scatter', False)
    vector_layout = profile.get('chunk_layout_vector4', False)
    compact_owner = profile.get('compact_owner_prefill', False)
    require(not wide_owner or (compact_owner and 'wide_owner_prefill' in backend and
            profile.get('maximum_owner_prefill_chunk')==512 and bound.get('owner_adaptive') is True),
            'wide owner profile differs from native preparation')
    require(type(compact_owner) is bool and (not compact_owner or ('compact_owner_prefill' in backend
            and profile.get('expected_compact_prefill_family_count') == (17 if wide_owner else 14))), 'compact owner policy differs from backend')
    require(counts['native_compact_prefill_prepared'] == int(compact_owner), 'missing/unexpected compact family inventory')
    if compact_owner:
        require(next(i for i,e in enumerate(events) if e['event'] == 'native_compact_prefill_prepared') <
                next(i for i,e in enumerate(events) if e['event'] == 'ready'), 'compact families prepared after readiness')
    require(type(vector_layout) is bool and vector_layout == ('chunk_layout_vector4' in backend)
            and (not vector_layout or profile.get('expected_chunk_layout_vector_dispatches') == 256),
            'vector layout policy differs from model/backend')
    require(type(kv_scatter) is bool and kv_scatter == ('chunk_kv_scatter_contiguous' in backend)
            and (not kv_scatter or profile.get('expected_chunk_kv_scatter_dispatches') == 64),
            'contiguous KV update policy differs from model/backend')
    require(type(mistral_reuse) is bool and mistral_reuse == ('mistral_decode_reuse_four_rows' in backend),
            'Mistral four-row decode reuse policy differs from backend')
    require(type(active_requests) is bool and active_requests == ('chunk_attention_active_rows' in backend)
            and (not active_requests or matrix_attention), 'physical request schedule differs from backend')
    packed = profile.get('mistral_packed_prefill', False)
    require(type(packed) is bool and packed == ('mistral_packed_prefill_optional' in backend)
            and (not packed or (profile.get('packed_prefill_min_rows')==128
                and profile.get('packed_prefill_plans')==3 and profile.get('packed_prefill_scratch_bytes')==14680064
                and {'packed_prefill_min_rows=128','packed_prefill_scratch_bytes=14680064'}<=backend
                and 'mistral_streamed_prefill_optional' not in backend)), 'packed compiled profile differs')
    require(counts['native_packed_prefill_prepared']==int(packed), 'missing/unexpected packed preparation')
    if packed:
        prepared=next(e for e in events if e['event']=='native_packed_prefill_prepared')
        require((prepared['plans'],prepared['scratch_buffers'],prepared['scratch_bytes'])==(3,1,14680064), 'incorrect packed workspace')
        require(events.index(prepared)<events.index(ready), 'packed plans prepared after ready')
    reuse64 = profile.get('require_reuse64', False)
    reuse128 = profile.get('require_reuse128', False)
    require(type(reuse128) is bool and reuse128 == ('mpp_affine_reuse128_optional' in backend)
            and (not reuse128 or (reuse64 and profile.get('reuse128_min_rows')==128
                and profile.get('reuse128_threadgroup_threads')==256 and profile.get('reuse128_even64_tiles') is True
                and {'mpp_affine_reuse128_min_rows=128','mpp_affine_reuse128_threads=256',
                     'mpp_affine_reuse128_even64_tiles=1'}<=backend)), 'cooperative128 compiled policy differs')
    require(type(reuse64) is bool and reuse64 == ('mpp_affine_reuse64_optional' in backend),
            'cooperative64 policy differs from backend')
    if reuse64:
        threshold = profile.get('reuse64_min_outputs')
        require(type(threshold) is int and threshold in (4096, 5120)
                and f'mpp_affine_reuse64_min_outputs={threshold}' in backend
                and profile.get('reuse64_relaxed') is False and 'mpp_affine_reuse64_relaxed=0' in backend
                and profile.get('expected_body_reuse64_projections') == (160 if threshold == 4096 else 64),
                'incorrect compiled cooperative64 profile')
    if prefix_policy:
        require(profile.get('native_cached_chunks') is True and 'native_cached_prefill_scheduler' in backend,
                'native cached-progress extension/profile missing')
    require(matrix_attention == ('chunk_attention_mps' in backend), 'attention policy differs from backend')
    if matrix_attention:
        require(profile.get('chunk_attention_buckets') is True and profile.get('chunk_attention_index32') is True
                and profile['expected_chunk_attention_dispatches'] == 32, 'incorrect attention profile')
        require(counts['native_chunk_attention_mps_prepared'] == counts['native_chunk_attention_buckets_prepared'] == 1,
                'missing prepared attention inventory')
    bulk_enabled=profile.get('native_leased_bulk_chunks',0)==16
    kv_enabled=profile['native_bulk_kv'];require(type(kv_enabled) is bool and (not kv_enabled or bulk_enabled),'wrong K/V profile')
    qo_enabled=profile['native_bulk_qo'];require(type(qo_enabled) is bool and (not qo_enabled or kv_enabled),'wrong Q/O profile')
    require(profile.get('native_cached_chunks') is bulk_enabled and profile['prefix_entries']==profile['prefix_host_bytes']==0,'incorrect cold tracking/retention profile')
    require(counts['native_bulk_prefill_prepared']==counts['native_tracking_only_prepared']==int(bulk_enabled),'missing/unexpected native bulk/tracking preparation')
    if bulk_enabled:
        preparation=next(e for e in events if e['event']=='native_bulk_prefill_prepared')
        require(preparation==dict(event='native_bulk_prefill_prepared',maximum_chunks=16,chunk_tokens=128,physical_rows=4,kv_enabled=profile['native_bulk_kv'],qo_enabled=qo_enabled),'wrong native group bound')
        require(events.index(preparation)<events.index(ready),'bulk prepared after readiness')
    for e in events:
        if e['event'] in ('native_tracking_only_prepared','native_prefix_stats'):
            st=e['stats'];require(st['retention_enabled'] is False and all(st[k]==0 for k in ['entries','accounted_host_bytes','hits','dense_capture_copied_bytes','dense_capture_compared_bytes']),'retained or restored a cold prefix')
    bulk_batches=0;bulk_pending=False;bulk_current=False;bulk_tickets=set()
    leases, rows, epochs, progress = {}, {}, [0]*4, {}
    chunks, tokens, completed = collections.defaultdict(list), collections.defaultdict(list), {}
    cached_prefixes,token_cached = {},{}
    stage_counts = collections.Counter()
    profile_counts = collections.Counter()
    pending = None
    pending_fresh = False
    pool_expected = None
    fresh_prefill_batches = 0
    attention_checks, attention_expected = set(), None
    maximum_active = 0
    reuse64_expected = None
    last_prefill_active = None
    requests = []
    scatter_expected = None
    layout_expected = None
    compact_expected = None
    compact_rows = 0
    physical_batch = 4
    current_chunk = maximum_chunk
    tick_batches = []
    tick_decode_tokens = 0
    for e in events:
        kind = e['event']
        if 'memory' in e:
            m = e['memory']
            require(0 <= m['metal_allocated_bytes'] <= m['metal_recommended_working_set_bytes']-256*1024*1024,
                    'sampled working-set bound exceeded')
        if kind == 'native_profile_tick':
            require(wide_owner and not tick_batches and pending is None, 'unfinished/overlapping native profile tick')
            tick_batches = list(e['batches'])
            stages = [b['stage'] for b in tick_batches]
            require(stages in (['decode'],['prefill'],['decode','prefill']), 'invalid native tick stage order')
            tick_decode_tokens = sum(len(b['requests']) for b in tick_batches if b['stage']=='decode')
            total = tick_decode_tokens + sum(r['prefill_tokens'] for b in tick_batches if b['stage']=='prefill' for r in b['requests'])
            require(total == e['token_count'] and 0<=total<=512, 'native profile tick exceeds global token budget')
        elif kind == 'native_batch':
            stage = e['stage']
            current_chunk = e.get('prefill_chunk',maximum_chunk) if stage=='prefill' else 1
            if wide_owner:
                require(bool(tick_batches), 'native batch missing prepared tick')
                planned = tick_batches.pop(0)
                require((e['stage'],e['requests'],e['active_rows'])==(planned['stage'],planned['requests'],planned['active_rows'])
                        and current_chunk==(planned['prefill_chunk'] if stage=='prefill' else 1), 'native batch differs from prepared tick')
                if stage=='prefill':
                    left = [r['prompt_tokens']-r['prefill_offset'] for r in e['requests']]
                    live = sorted(r['row'] for r,n in zip(e['requests'],left) if n)
                    longest = max(left);wanted = 2
                    if longest<=128:
                        while wanted<longest:wanted*=2
                    else:
                        while wanted*2<=min(longest,512):wanted*=2
                    expected_chunk = 0
                    while wanted>=2:
                        span = len(live)==1 or (len(live)==2 and live[1]==live[0]+1)
                        admissible = wanted<=128 or (span and len(live)*wanted<=512)
                        if admissible and sum(min(n,wanted) for n in left)<=512-tick_decode_tokens:
                            expected_chunk=wanted;break
                        wanted//=2
                    require(current_chunk==expected_chunk and current_chunk in (2,4,8,16,32,64,128,256,512),
                            'wrong prepared owner-prefill shape selection')
            else:require(current_chunk==(maximum_chunk if stage=='prefill' else 1), 'unexpected variable legacy chunk')
            require(compact_expected is None, 'batch missing compact row route')
            require(layout_expected is None, 'batch missing vector layout coverage')
            require(scatter_expected is None, 'batch missing contiguous KV coverage')
            require(pending is None, 'batch missing affine coverage')
            require(reuse64_expected is None, 'batch missing cooperative64 coverage')
            stage = e['stage']
            require(stage in ('prefill', 'decode'), 'unexpected schedule stage')
            requests = e['requests']
            physical = [r['row'] for r in requests]
            require(requests and len(set(physical)) == len(physical)
                    and all(type(r) is int and 0 <= r < 4 for r in physical)
                    and e['active_rows'] == [int(r in physical) for r in range(4)], 'invalid active rows')
            maximum_active = max(maximum_active, len(requests))
            if stage == 'prefill':
                bulk_current=any(r['prefill_tokens']>128 for r in requests)
                if bulk_current:
                    require(bulk_enabled and current_chunk==128 and len(requests)==4 and all(r['prefill_tokens']==2048 and r['generated_tokens']==0 and r['position_delta']==0 and r['context_length']==r['prefill_offset']+2048 for r in requests),'invalid native coalesced group/fairness contract')
                    require(sum(r['prefill_tokens'] for r in requests)==8192,'grouped budget exceeded')
                else:require(sum(r['prefill_tokens'] for r in requests)<=512,'ordinary physical prefill budget exceeded')
            else:bulk_current=False
            for r in requests:
                identity, row, epoch = r['id'], r['row'], r['epoch']
                cached = r.get('cached_tokens',0)
                require(type(cached) is int and 0 <= cached <= r['prompt_tokens'] and (prefix_policy or cached == 0),
                        'invalid native restored prefix')
                if identity not in leases:
                    require(stage == 'prefill' and r['prefill_offset'] == cached
                            and row not in rows and epoch > epochs[row], 'stale/overlapping row lease')
                    leases[identity] = (row, epoch)
                    rows[row], epochs[row], progress[identity] = identity, epoch, cached
                    cached_prefixes[identity] = cached
                require(cached_prefixes[identity] == cached, 'native reused prefix changed')
                require(leases[identity] == (row, epoch) and rows.get(row) == identity,
                        'row identity/epoch changed')
                if stage == 'prefill':
                    cached_head = prefix_policy and r['prefill_tokens'] == 0 and cached == r['prompt_tokens']
                    require((0 < r['prefill_tokens'] <= (2048 if bulk_current else current_chunk) or cached_head) and r['prefill_offset'] == progress[identity]
                            and r['slot'] == r['prefill_offset'] and r['generated_tokens'] == 0
                            and r['prefill_offset']+r['prefill_tokens'] <= r['prompt_tokens'],
                            'invalid prefill position/chunk')
                    progress[identity] += r['prefill_tokens']
                    chunks[identity].append((r['prefill_offset'], r['prefill_tokens'],
                                             progress[identity] == r['prompt_tokens']))
                else:
                    require(progress[identity] >= r['prompt_tokens']
                            and r['slot'] == progress[identity]
                            and r['generated_tokens'] == r['slot']-r['prompt_tokens']+1,
                            'invalid decode position')
                    progress[identity] += 1
            pending = stage
            pending_fresh = stage != 'prefill' or any(r['prefill_tokens'] for r in requests)
            if stage == 'prefill' and pending_fresh:fresh_prefill_batches += 1
            if kv_scatter and pending_fresh:scatter_expected = (stage,64 if stage == 'prefill' else 0)
            compact_rows, compact_first = 0, 4
            if compact_owner and pending_fresh:
                live = sorted(r['row'] for r in requests if stage != 'prefill' or r['prefill_tokens'] > 0)
                if stage == 'prefill' and len(live) in (1,2) and len(live)*current_chunk>4 and (len(live) == 1 or live[1] == live[0]+1):
                    compact_rows, compact_first = len(live), live[0]
                compact_expected = (stage,compact_rows,compact_first,current_chunk if stage == 'prefill' else 1)
            physical_batch = compact_rows or 4
            if vector_layout and pending_fresh:layout_expected = (stage,256+int(compact_rows>0) if stage == 'prefill' else 0)
            if matrix_attention and pending_fresh:
                active = sum(r['prefill_tokens'] for r in requests) if stage == 'prefill' else len(requests)
                tail = stage == 'prefill' and active <= 8 and not half_attention
                extent = 0
                if stage == 'prefill' and not tail:
                    visible = max(r['prefill_offset']+r['prefill_tokens'] for r in requests)
                    extent = 128
                    while extent < visible:
                        extent *= 2
                active_owners = sum(r['prefill_tokens'] > 0 for r in requests) if stage == 'prefill' else len(requests)
                attention_expected = (stage, active, tail, extent, active_owners)
                attention_checks = set()
            stage_counts[stage] += 1
            if bulk_current:bulk_pending=True
        elif kind=='native_bulk_prefill_execution':
            require(bulk_enabled and bulk_pending and pending=='prefill' and pending_fresh and len(requests)==4,'bulk execution not bound to its current native plan')
            st=e['stats'];require(st['leased'] is True and st['enabled'] is True and st['chunks']==st['bound_chunks']==16 and st['rows']==8192 and st['workspace_bytes']==1909522432 and st['appended']==[2048]*4 and st['packed_calls']==(0 if qo_enabled else 1024) and st['bulk_calls']==(224 if qo_enabled else 160 if kv_enabled else 96) and st['qo_bulk_enabled'] is qo_enabled and st['qo_bulk_calls']==(64 if qo_enabled else 0) and st['qo_input_ops_skipped']==(2560 if qo_enabled else 0) and st['kv_bulk_calls']==(64 if kv_enabled else 0) and st['kv_bulk_enabled'] is kv_enabled and st['silu_half_calls']==32,'wrong actual grouped native route')
            require(type(e['ticket']) is int and e['ticket']>0 and e['ticket'] not in bulk_tickets,'bulk execution ticket reused');bulk_tickets.add(e['ticket'])
            require(e['ended_monotonic']>e['began_monotonic'] and e['run_wall_ms']>0 and abs(e['run_wall_ms']-(e['ended_monotonic']-e['began_monotonic'])*1000)<1e-6 and 0<st['gpu_ms']<=st['body_wall_ms']<=e['run_wall_ms'],'invalid native group times')
            bulk_batches+=1;bulk_pending=False;pending=None;compact_expected=None;layout_expected=None;scatter_expected=None;attention_expected=None;attention_checks=set()

        elif kind == 'native_prefill_pool':
            require(hidden_pool and pool_expected==(e['stage'],e['bindings']), 'incorrect hidden pool binding route')
            pool_expected=None
        elif kind == 'native_compact_prefill_prepared':
            require(compact_owner and e['families'] == (17 if wide_owner else 14)
                    and (not wide_owner or e.get('maximum_chunk')==512), 'incorrect compact family inventory')
        elif kind == 'native_compact_prefill':
            require(compact_owner and compact_expected == (e['stage'],e['rows'],e['first'],e['chunk']), 'incorrect compact physical owner route')
            compact_expected = None
        elif kind == 'native_chunk_attention_mps_prepared':
            require(matrix_attention and e['plan_count'] == profile['expected_attention_mps_plan_count']
                    and e['scratch_bytes'] == profile['expected_attention_mps_scratch_bytes'],
                    'changed prepared attention storage')
        elif kind == 'native_chunk_attention_buckets_prepared':
            require(matrix_attention and e['bucket_count'] == profile['expected_attention_mps_bucket_count'],
                    'changed prepared attention buckets')
        elif kind in ('native_chunk_half_attention', 'native_chunk_flash_attention', 'native_chunk_attention_mps', 'native_chunk_attention_index32', 'native_chunk_attention_bucket',
                      'native_chunk_attention_request_products'):
            require(matrix_attention and pending is not None and pending_fresh and kind not in attention_checks,
                    'unexpected/duplicate attention dispatch event')
            attention_checks.add(kind)
            stage, active, tail, extent, active_owners = attention_expected
            require(e['stage'] == stage, 'attention stage/count mismatch')
            if kind != 'native_chunk_attention_request_products':
                require(e['active_tokens'] == active, 'attention stage/count mismatch')
            expected = 32 if stage == 'prefill' else 0
            if kind == 'native_chunk_attention_request_products':
                products = (0 if tail or flash_attention or half_attention else expected)*active_owners*8*4
                skipped = (0 if tail or flash_attention or half_attention else expected)*(physical_batch-active_owners)*8*4
                require(active_requests and e['matrices'] == products and e['skipped_matrices'] == skipped
                        and e['active_request_rows'] == active_owners, 'incorrect physical request matrix ranges')
            elif kind == 'native_chunk_attention_mps':
                require(e['dispatches'] == (0 if tail or flash_attention or half_attention else expected)
                        and e['plan_count'] == profile['expected_attention_mps_plan_count']
                        and e['scratch_bytes'] == profile['expected_attention_mps_scratch_bytes'],
                        'incorrect attention matrix route/storage')
            elif kind == 'native_chunk_half_attention':
                require(half_attention and e['dispatches'] == expected, 'incorrect half attention dispatch')
            elif kind == 'native_chunk_flash_attention':
                require(flash_attention and e['dispatches'] == (0 if tail else expected), 'incorrect fused flash attention route')
            elif kind == 'native_chunk_attention_index32':
                require(e['dispatches'] == (32 if tail else 0), 'incorrect attention scalar-tail route')
            else:
                require(e['dispatches'] == expected and e['tail_dispatches'] == (32 if tail else 0)
                        and e['capacity'] == extent and e['unknown_controls'] == 0
                        and e['bucket_count'] == profile['expected_attention_mps_bucket_count'],
                        'incorrect prepared attention context/tail decision')
        elif kind == 'native_chunk_kv_scatter':
            require(kv_scatter and scatter_expected == (e['stage'],e['dispatches']),
                    'incorrect/unexpected contiguous KV dispatch coverage')
            scatter_expected = None
        elif kind == 'native_chunk_layout_vector':
            require(vector_layout and layout_expected == (e['stage'],e['dispatches']),
                    'incorrect/unexpected vector layout dispatch coverage')
            layout_expected = None
        elif kind == 'native_affine_profile':
            require(pool_expected is None, 'missing pool event before next affine stage')
            if hidden_pool and e['stage'] in ('prefill','decode'):pool_expected=(e['stage'],32 if e['stage']=='prefill' else 0)
            stage = e['stage']
            require(stage in ('prefill', 'decode', 'head'), 'unknown affine stage')
            if stage != 'head':
                require(pending == stage and pending_fresh, 'affine stage mismatch')
                require(not matrix_attention or len(attention_checks) == 3+int(active_requests)+int(flash_attention)+int(half_attention), 'missing attention route coverage')
                pending = None
            elif pending == 'prefill' and not pending_fresh:
                pending = None # Restored final hidden state; no prefill body ran.
            require(e['affine_dispatches'] == dict(prefill=224, decode=225, head=1)[stage]
                    and e['misses'] == 0 and 0 < e['pipeline_count'] <= 512
                    and e['index_bits'] == 32, 'incomplete affine dispatch coverage')
            if mistral_reuse:
                active = (sum(r['prefill_tokens'] for r in requests) if stage == 'prefill' else
                          len(requests) if stage == 'decode' else
                          sum(r['prefill_offset']+r['prefill_tokens'] == r['prompt_tokens'] for r in requests))
                require(e['active_rows'] == active and e['physical_rows'] == (physical_batch*current_chunk if stage == 'prefill' else 4),
                        'Mistral reuse extent differs from native row leases')
                expected_reuse = (161 if stage == 'decode' else 1) if active == 4 and stage != 'prefill' else 0
                require(e['mistral_decode_reuse_dispatches'] == expected_reuse, 'incorrect Mistral reuse projection coverage')
            else:
                require(e.get('mistral_decode_reuse_dispatches',0) == 0, 'unexpected Mistral reuse dispatch')
            if stage=='prefill':
                last_prefill_active=sum(r['prefill_tokens'] for r in requests)
                require(not packed or e['active_rows']==last_prefill_active, 'packed extent differs from native plan')
            profile_counts[stage] += 1
        elif kind in ('native_mpp', 'native_indexed'):
            expected = 224 if kind == 'native_mpp' or e['active_tokens'] < physical_batch*current_chunk else 0
            require(e['affine_dispatches'] == expected, 'incorrect prefill projection route')
            if packed and kind=='native_mpp':
                selected=160 if last_prefill_active>=128 else 0
                require(e['packed_prefill_dispatches']==selected and e['legacy_mpp_dispatches']==224-selected,
                        'incorrect packed/legacy MPP accounting')
            if packed and kind=='native_indexed':
                require(e['active_tokens']==last_prefill_active, 'packed indexed extent differs')
            if kind == 'native_indexed':
                require(0 < e['active_tokens'] <= physical_batch*current_chunk and e['physical_rows'] == physical_batch*current_chunk,
                        'invalid indexed prefill extent')
                if reuse64:
                    require(reuse64_expected is None, 'missing prior cooperative64 event')
                    reuse64_expected = (e['active_tokens'], e['physical_rows'])
        elif kind in ('native_reuse64','native_cooperative_affine'):
            require(reuse64 and reuse64_expected == (e['logical_rows'], e['physical_rows']),
                    'cooperative64 extent differs from active native rows')
            expected=profile['expected_body_reuse64_projections'] if e['logical_rows']>=64 else 0
            require(e['affine_dispatches']==expected and (kind=='native_cooperative_affine')==reuse128,
                    'incorrect cooperative projection coverage')
            if reuse128:
                packed_selected=expected if packed and e['logical_rows']>=128 else 0
                selected=expected-packed_selected if e['logical_rows']>=128 and ((e['logical_rows']+63)//64)%2==0 else 0
                require(e['reuse128_dispatches']==selected and e['reuse64_dispatches']==expected-selected-packed_selected
                        and e['reuse128_threads']==256 and e.get('packed_dispatches',0)==packed_selected,
                        'incorrect native64/128/packed tile selection')
            reuse64_expected = None
        elif kind == 'native_indexed_decode':
            require(e['stage'] in ('decode', 'head') and 0 < e['active_rows'] <= 4
                    and e['physical_rows'] == 4, 'invalid indexed decode extent')
            expected = (225 if e['stage'] == 'decode' else 1) if e['active_rows'] < 4 else 0
            require(e['affine_dispatches'] == expected, 'incorrect indexed decode route')
        elif kind == 'native_prefill_chunk':
            identity = e['native_request']
            require(identity in leases and leases[identity] == (e['row'], e['epoch'])
                    and chunks[identity], 'unknown prefill acknowledgement')
            require(chunks[identity].pop(0) == (e['offset'], e['tokens'], e['final']),
                    'incorrect prefill acknowledgement')
        elif kind == 'token':
            require(not bulk_pending,'token published before bulk execution completed')
            require(e['id'] not in completed, 'token after completion')
            ts = tokens[e['id']]
            cached = e['cached_tokens']
            require(type(cached) is int and 0 <= cached <= e['prompt_tokens'] and (prefix_policy or cached == 0),
                    'invalid token cache count')
            require(e['id'] not in token_cached or token_cached[e['id']] == cached, 'token cache count changed')
            token_cached[e['id']] = cached
            require(e['sampled_tokens'] == len(ts)+1,
                    'incorrect incremental token count/cache policy')
            ts.append(e['token_id'])
        elif kind == 'completed':
            key, identity = e['id'], e['native_request']
            require(key not in completed and identity in leases
                    and leases[identity] == (e['native_row'], e['native_epoch'])
                    and rows.get(e['native_row']) == identity, 'invalid completion lease')
            ids = e['generated_ids']
            require(ids == tokens[key] and ids and all(type(i) is int and 0 <= i < 32768 for i in ids)
                    and not any(i == 2 for i in ids[:-1]), 'invalid output IDs/EOS')
            require(e['finish_reason'] == ('stop' if ids[-1] == 2 else 'length')
                    and e['cached_tokens'] == cached_prefixes[identity] == token_cached[key] and e['profile_id'] == profile['profile_id']
                    and e['generation_mode'] == 'native_metal_continuous_batch'
                    and progress[identity] == len(e['prompt_ids'])+len(ids)-1,
                    'invalid completion state')
            completed[key] = e
            del rows[e['native_row']]
    generic_prefill_batches=fresh_prefill_batches-bulk_batches
    require(not bulk_pending and counts['native_bulk_prefill_execution']==bulk_batches and (bulk_batches>=1 if bulk_enabled else bulk_batches==0),'missing/unexpected grouped execution')
    require(not tick_batches and compact_expected is None and not rows and pending is None and reuse64_expected is None and scatter_expected is None and layout_expected is None and all(not c for c in chunks.values()), 'unfinished native work')
    require(pool_expected is None, 'missing final hidden pool event')
    require(counts['native_prefill_pool']==(generic_prefill_batches+stage_counts['decode'] if hidden_pool else 0), 'missing/unexpected hidden pool route')
    require(counts['native_compact_prefill'] == (generic_prefill_batches+stage_counts['decode'] if compact_owner else 0), 'missing compact execution route')
    require(counts['native_chunk_flash_attention'] == (generic_prefill_batches+stage_counts['decode'] if flash_attention else 0), 'missing fused attention stage event')
    require(counts['native_chunk_half_attention']==generic_prefill_batches+stage_counts['decode'], 'missing half route event')
    require(wide_owner or counts['native_profile_tick']==0, 'unexpected adaptive profile tick')
    require(counts['native_chunk_layout_vector'] == (generic_prefill_batches+stage_counts['decode'] if vector_layout else 0),
            'missing vector layout stage event')
    require(counts['native_chunk_kv_scatter'] == (generic_prefill_batches+stage_counts['decode'] if kv_scatter else 0),
            'missing contiguous KV stage event')
    require(counts['native_reuse64'] == (generic_prefill_batches if reuse64 and not reuse128 else 0)
            and counts['native_cooperative_affine'] == (generic_prefill_batches if reuse128 else 0),
            'missing/unexpected cooperative64 event')
    require(profile_counts['prefill'] == generic_prefill_batches == counts['native_mpp'] == counts['native_indexed']
            and profile_counts['decode'] == stage_counts['decode']
            and counts['native_indexed_decode'] == profile_counts['decode']+profile_counts['head'],
            'missing projection event')
    require(maximum_active == required_active_rows and len(completed) == len(clients) == expected_requests, 'missing B4/request coverage')
    require(set(completed) == {c['id'] for c in clients}, 'HTTP/native identity mismatch')
    require(set(tokens) == set(completed), 'orphan native token stream')
    details = []
    for c in clients:
        e, result, body = completed[c['id']], c['result'], c['body']
        require(c['sse_done'] == 1 and c['sse_ids'] == e['generated_ids']
                and c['final_prompt_ids'] == e['prompt_ids'] == c['expected_prompt_ids']
                and c['final_response_ids'] == e['generated_ids'], 'HTTP/frozen/native token mismatch')
        require(result['status'] == 'completed' and result['text'] == c['sse_text'] == e['text']
                and result['generated_tokens'] == len(e['generated_ids'])
                and result['prompt_tokens'] == len(e['prompt_ids']) and result['cached_prompt_tokens'] == e['cached_tokens']
                and c['stream_cached_counts'] == [e['cached_tokens']]
                and result['finish_reason'] == c['sse_finish'] == e['finish_reason'],
                'HTTP result mismatch')
        require(len(e['generated_ids']) <= body['max_tokens']
                and (e['finish_reason'] == 'stop' or len(e['generated_ids']) == body['max_tokens'])
                and len(e['prompt_ids'])+body['max_tokens'] <= 4096, 'generation budget mismatch')
        control = controls.get(c['plan'])
        if control:
            expected = control['generated_ids'][:body['max_tokens']] if control_prefixes else control['generated_ids']
            require(control['status'] == 'completed' and control['prompt_ids'] == e['prompt_ids']
                    and expected == e['generated_ids']
                    and (control_prefixes or control['text'] == e['text']),
                    'frozen native coding control divergence')
        details.append(dict(plan=c['plan'], request=c['id'], native_request=e['native_request'],
                            row=e['native_row'], epoch=e['native_epoch'], output_tokens=len(e['generated_ids']),
                            exact_native_coding_control=True if control else None))
    close = events[-1]['memory']
    require(all(close[k] == 0 for k in OWNERS) and 0 <= close['metal_allocated_bytes'] <= 4*1024*1024,
            'native cleanup failed')
    return dict(bulk_prefill_batches=bulk_batches,bulk_tickets=sorted(bulk_tickets),requests=details, maximum_active_rows=maximum_active, event_counts=dict(counts),
                head_only_prefill_batches=stage_counts['prefill']-fresh_prefill_batches,
                close_allocation_bytes=close['metal_allocated_bytes'])


def load_client(prefix, plan, body, expected_prompt_ids, read, raw):
    """Bind a saved request/result to its raw stream; caller records file hashes."""
    require(read(prefix.with_suffix('.request.json')) == body, 'changed HTTP request')
    wire = [json.loads(line) for line in raw(prefix.with_suffix('.events.jsonl')).splitlines()]
    require(wire and wire[-1]['data'] == '[DONE]', 'unterminated SSE')
    data = [json.loads(w['data']) for w in wire if w['data'] != '[DONE]']
    identities = {d['id'] for d in data}
    require(len(identities) == 1, 'invalid stream identity')
    finals = [d for d in data if 'response_token_ids' in d]
    require(len(finals) == 1, 'missing/duplicate final token arrays')
    return dict(plan=plan, id=next(iter(identities)), body=body,
        result=read(prefix.with_suffix('.result.json')), expected_prompt_ids=expected_prompt_ids,
        sse_ids=[d['token_id'] for d in data if 'token_id' in d],
        sse_done=sum(w['data'] == '[DONE]' for w in wire),
        sse_text=''.join(c.get('delta', {}).get('content', '') for d in data for c in d.get('choices', [])),
        sse_finish=finals[0]['choices'][0]['finish_reason'],
        stream_cached_counts=sorted({(d['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0) for d in data if d.get('usage')}),
        final_prompt_ids=finals[0]['prompt_token_ids'], final_response_ids=finals[0]['response_token_ids'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--coding-control', type=Path, required=True)
    parser.add_argument('--control-provenance', type=Path, required=True)
    parser.add_argument('--control-provenance-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = {}

    def raw(path):
        data = path.read_bytes()
        files[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return data

    def read(path):
        return json.loads(raw(path))

    provenance = read(args.control_provenance)
    require(files[str(args.control_provenance.resolve())] == args.control_provenance_sha256,
            'changed control provenance')
    fixture = read(args.fixture)
    controls = {c['id']: read(args.coding_control/c['id']/'request-0.result.json')
                for c in fixture['cases'] if c['kind'] == 'coding'}
    for path, digest in files.items():
        if path != str(args.control_provenance.resolve()):
            require(provenance['files'].get(path) == digest, 'unfrozen control/fixture: '+path)
    root = args.admission
    require(read(root/'campaign/status.json')['status'] == 'completed_collection_not_automatic_qualification',
            'incomplete campaign')
    require(read(root/'accounting-audit.json')['issues'] == [], 'failed accounting audit')
    manifest, profile = read(root/'manifest.json'), read(root/'profile.json')
    events = [json.loads(json.loads(line)['line']) for line in raw(root/'server-1/worker-stdout.jsonl').splitlines()]
    exit_status = read(root/'server-1/worker-exit.json')
    clients = []
    for plan in manifest['plans']:
        bodies, workload = read(Path(plan['cases'])), read(Path(plan['workload']))
        require(files[str(Path(plan['cases']).resolve())] == plan['cases_sha256'] == workload['cases_sha256'],
                'changed frozen request bodies')
        require(len(bodies) == len(workload['requests']), 'invalid wave length')
        for i, (body, expected) in enumerate(zip(bodies, workload['requests'])):
            prefix = root/'campaign/cohort-1'/(plan['id']+'-measured-0')/('request-'+str(i))
            clients.append(load_client(prefix, plan['id'], body, expected['token_ids'], read, raw))
    if profile.get('mistral_packed_prefill'):
        inventory=read(Path(profile['memory_inventory_path']))
        refs=[x for x in inventory['references'] if x['role']=='packed_prefill_half_input']
        require(len(refs)==1 and inventory['buffers'][refs[0]['buffer_id']]['length_bytes']==14680064,
                'saved packed workspace inventory differs')
    report = validate(events, clients, controls, profile, exit_status)
    rejected = []
    corruptions = ['wrong-bound', 'stale-lease', 'wrong-slot', 'oversized-chunk', 'missing-projection',
                   'wrong-token', 'changed-prompt', 'wrong-stream', 'wrong-control', 'cache-hit', 'live-owner']
    if profile.get('chunk_attention_mps'):
        corruptions += ['changed-attention-storage', 'wrong-context', 'wrong-tail', 'missing-attention-route']
    if profile.get('require_reuse64'):
        corruptions += ['wrong-reuse-threshold', 'wrong-reuse-rows', 'wrong-reuse-count', 'missing-reuse-event']
    if profile.get('chunk_attention_active_rows'):
        corruptions += ['wrong-request-matrices', 'wrong-request-owners', 'missing-request-schedule']
    if profile.get('mistral_decode_reuse'):
        corruptions += ['wrong-mistral-reuse-policy', 'wrong-mistral-reuse-count', 'wrong-mistral-reuse-owners']
    if profile.get('chunk_kv_scatter'):
        corruptions += ['wrong-kv-scatter-policy', 'wrong-kv-scatter-count', 'missing-kv-scatter-event']
    if profile.get('chunk_layout_vector4'):
        corruptions += ['wrong-vector-layout-policy', 'wrong-vector-layout-count', 'missing-vector-layout-event']
    if profile.get('compact_owner_prefill'):
        corruptions += ['wrong-compact-policy','wrong-compact-inventory','wrong-compact-span','missing-compact-route','wrong-compact-footprint']
    if profile.get('prefill_alias_groups'):
        corruptions += ['alias-policy','alias-layout','missing-alias-preparation','alias-lifetime']
    if profile.get('prefill_hidden_pool'):
        corruptions += ['pool-policy','pool-bindings','missing-pool-route']
    if profile.get('mistral_grouped_value_context'):
        corruptions += ['grouped-value-policy']
    if profile.get('chunk_flash_attention'):
        corruptions += ['wrong-flash-policy','wrong-flash-dispatch','missing-flash-route']
    if profile.get('wide_owner_prefill'):
        corruptions += ['wrong-wide-policy','wrong-wide-inventory','wrong-wide-bound',
                        'missing-profile-tick','wrong-tick-budget','wrong-prepared-chunk','wrong-tick-order']
    if profile.get('chunk_attention_scratch'):
        corruptions += ['wrong-attention-scratch-policy']
    if profile.get('require_reuse128'):
        # The new event carries both real kernel counts; preserve total work
        # while replacing128 by64 to exercise the independent policy decision.
        corruptions=[c for c in corruptions if c not in ('wrong-reuse-rows','wrong-reuse-count','missing-reuse-event')]
        corruptions += ['wrong-reuse128-policy','wrong-reuse128-route','wrong-reuse128-threads','missing-cooperative-event']
    if profile.get('mistral_packed_prefill'):
        corruptions=[c for c in corruptions if c!='wrong-reuse128-route']
        corruptions+=['wrong-packed-policy','wrong-packed-workspace','missing-packed-preparation','wrong-packed-route','wrong-packed-mpp']
    for label in corruptions:
        es, cs, ct = copy.deepcopy(events), copy.deepcopy(clients), copy.deepcopy(controls)
        if label == 'wrong-packed-policy':
            next(e for e in es if e['event']=='ready')['backend']=next(e for e in es if e['event']=='ready')['backend'].replace('packed_prefill_min_rows=128','packed_prefill_min_rows=64')
        elif label == 'wrong-packed-workspace':next(e for e in es if e['event']=='native_packed_prefill_prepared')['scratch_bytes']=4
        elif label == 'missing-packed-preparation':es.remove(next(e for e in es if e['event']=='native_packed_prefill_prepared'))
        elif label == 'wrong-packed-route':
            e=next(e for e in es if e['event']=='native_cooperative_affine' and e['packed_dispatches'])
            e['reuse128_dispatches']+=e['packed_dispatches'];e['packed_dispatches']=0
        elif label == 'wrong-packed-mpp':
            e=next(e for e in es if e['event']=='native_mpp' and e['packed_prefill_dispatches'])
            e['legacy_mpp_dispatches']+=e['packed_prefill_dispatches'];e['packed_prefill_dispatches']=0
        elif label == 'wrong-bound':
            next(e for e in es if e['event'] == 'native_dense_schedule_bound')['global_token_budget'] = 1024
        elif label == 'wrong-attention-scratch-policy':
            ready = next(e for e in es if e['event'] == 'ready')
            ready['backend'] = ready['backend'].replace('chunk_attention_full_write_scratch','chunk_attention_full_write_scratch_BAD')
        elif label == 'wrong-reuse128-policy':
            ready=next(e for e in es if e['event']=='ready')
            ready['backend']=ready['backend'].replace('mpp_affine_reuse128_optional','mpp_affine_reuse128_optional_BAD')
        elif label == 'wrong-reuse128-route':
            e=next(e for e in es if e['event']=='native_cooperative_affine' and e['reuse128_dispatches'])
            e['reuse64_dispatches']+=e['reuse128_dispatches'];e['reuse128_dispatches']=0
        elif label == 'wrong-reuse128-threads':
            next(e for e in es if e['event']=='native_cooperative_affine')['reuse128_threads']=128
        elif label == 'missing-cooperative-event':es.remove(next(e for e in es if e['event']=='native_cooperative_affine'))
        elif label == 'wrong-wide-policy':
            ready = next(e for e in es if e['event'] == 'ready')
            ready['backend'] = ready['backend'].replace('wide_owner_prefill','wide_owner_prefill_BAD')
        elif label == 'wrong-wide-inventory':
            next(e for e in es if e['event'] == 'native_compact_prefill_prepared')['maximum_chunk'] = 128
        elif label == 'wrong-wide-bound':
            next(e for e in es if e['event'] == 'native_dense_schedule_bound')['owner_adaptive'] = False
        elif label == 'missing-profile-tick':
            es.remove(next(e for e in es if e['event'] == 'native_profile_tick'))
        elif label == 'wrong-tick-budget':
            next(e for e in es if e['event'] == 'native_profile_tick')['token_count'] = 513
        elif label == 'wrong-prepared-chunk':
            # Change both copies to test the independent policy calculation,
            # rather than merely testing equality of the tick/batch records.
            tick = next(e for e in es if e['event'] == 'native_profile_tick' and any(b['stage']=='prefill' for b in e['batches']))
            batch = next(b for b in tick['batches'] if b['stage']=='prefill')
            event = next(e for e in es if e['event']=='native_batch' and e['stage']=='prefill')
            batch['prefill_chunk'] = event['prefill_chunk'] = 2 if event['prefill_chunk']!=2 else 4
        elif label == 'wrong-tick-order':
            next(e for e in es if e['event'] == 'native_profile_tick')['batches'][0]['stage'] = 'invalid'
        elif label == 'stale-lease': next(e for e in es if e['event'] == 'native_batch')['requests'][0]['epoch'] = 0
        elif label == 'wrong-slot': next(e for e in es if e['event'] == 'native_batch')['requests'][0]['slot'] += 1
        elif label == 'oversized-chunk': next(e for e in es if e['event'] == 'native_batch')['requests'][0]['prefill_tokens'] = 513 if profile.get('wide_owner_prefill') else 129
        elif label == 'missing-projection': next(e for e in es if e['event'] == 'native_affine_profile')['affine_dispatches'] = 0
        elif label == 'wrong-token': next(e for e in es if e['event'] == 'completed')['generated_ids'][0] ^= 1
        elif label == 'changed-prompt': cs[0]['expected_prompt_ids'][0] ^= 1
        elif label == 'wrong-stream': cs[0]['final_response_ids'][0] ^= 1
        elif label == 'wrong-control': next(iter(ct.values()))['generated_ids'][0] ^= 1
        elif label == 'cache-hit': next(e for e in es if e['event'] == 'completed')['cached_tokens'] = 1
        elif label == 'changed-attention-storage': next(e for e in es if e['event'] == 'native_chunk_attention_mps')['scratch_bytes'] += 1
        elif label == 'wrong-context': next(e for e in es if e['event'] == 'native_chunk_attention_bucket')['capacity'] = 4096
        elif label == 'wrong-tail': next(e for e in es if e['event'] == 'native_chunk_attention_bucket')['tail_dispatches'] = 32
        elif label == 'missing-attention-route': es.remove(next(e for e in es if e['event'] == 'native_chunk_attention_mps'))
        elif label == 'wrong-reuse-threshold':
            ready = next(e for e in es if e['event'] == 'ready')
            threshold = str(profile['reuse64_min_outputs'])
            ready['backend'] = ready['backend'].replace('mpp_affine_reuse64_min_outputs='+threshold,
                                                       'mpp_affine_reuse64_min_outputs=0')
        elif label == 'wrong-reuse-rows': next(e for e in es if e['event'] == 'native_reuse64')['logical_rows'] += 1
        elif label == 'wrong-reuse-count': next(e for e in es if e['event'] == 'native_reuse64' and e['affine_dispatches'])['affine_dispatches'] = 0
        elif label == 'missing-reuse-event': es.remove(next(e for e in es if e['event'] == 'native_reuse64'))
        elif label == 'wrong-request-matrices': next(e for e in es if e['event'] == 'native_chunk_attention_request_products')['matrices'] += 1
        elif label == 'wrong-request-owners': next(e for e in es if e['event'] == 'native_chunk_attention_request_products')['active_request_rows'] += 1
        elif label == 'missing-request-schedule': es.remove(next(e for e in es if e['event'] == 'native_chunk_attention_request_products'))
        elif label == 'wrong-mistral-reuse-policy':
            ready = next(e for e in es if e['event'] == 'ready')
            ready['backend'] = ready['backend'].replace('mistral_decode_reuse_four_rows','mistral_decode_reuse_four_rows_BAD')
        elif label == 'wrong-kv-scatter-policy':
            ready = next(e for e in es if e['event'] == 'ready')
            ready['backend'] = ready['backend'].replace('chunk_kv_scatter_contiguous','chunk_kv_scatter_contiguous_BAD')
        elif label == 'wrong-compact-policy':
            ready = next(e for e in es if e['event'] == 'ready');ready['backend'] = ready['backend'].replace('compact_owner_prefill','compact_owner_prefill_BAD')
        elif label == 'wrong-compact-inventory': next(e for e in es if e['event'] == 'native_compact_prefill_prepared')['families'] = 0
        elif label == 'wrong-compact-span': next(e for e in es if e['event'] == 'native_compact_prefill' and e['rows'])['first'] += 1
        elif label == 'missing-compact-route': es.remove(next(e for e in es if e['event'] == 'native_compact_prefill'))
        elif label == 'wrong-compact-footprint': next(e for e in es if e['event'] == 'native_affine_profile' and e['stage'] == 'prefill')['physical_rows'] *= 4
        elif label == 'wrong-flash-policy':
            ready = next(e for e in es if e['event'] == 'ready');ready['backend'] = ready['backend'].replace('chunk_flash_attention_fp32_operands','chunk_flash_attention_fp32_operands_BAD')
        elif label == 'wrong-flash-dispatch': next(e for e in es if e['event'] == 'native_chunk_flash_attention' and e['dispatches'])['dispatches'] = 0
        elif label == 'missing-flash-route': es.remove(next(e for e in es if e['event'] == 'native_chunk_flash_attention'))
        elif label == 'wrong-vector-layout-policy':
            ready = next(e for e in es if e['event'] == 'ready')
            ready['backend'] = ready['backend'].replace('chunk_layout_vector4','chunk_layout_vector4_BAD')
        elif label == 'wrong-vector-layout-count': next(e for e in es if e['event'] == 'native_chunk_layout_vector' and e['dispatches'])['dispatches'] = 0
        elif label == 'missing-vector-layout-event': es.remove(next(e for e in es if e['event'] == 'native_chunk_layout_vector'))
        elif label == 'wrong-kv-scatter-count': next(e for e in es if e['event'] == 'native_chunk_kv_scatter' and e['dispatches'])['dispatches'] = 0
        elif label == 'missing-kv-scatter-event': es.remove(next(e for e in es if e['event'] == 'native_chunk_kv_scatter'))
        elif label == 'wrong-mistral-reuse-count': next(e for e in es if e['event'] == 'native_affine_profile' and e['mistral_decode_reuse_dispatches'])['mistral_decode_reuse_dispatches'] = 0
        elif label == 'wrong-mistral-reuse-owners': next(e for e in es if e['event'] == 'native_affine_profile' and e['mistral_decode_reuse_dispatches'])['active_rows'] = 3
        elif label == 'alias-policy':next(e for e in es if e['event']=='ready')['backend']=next(e for e in es if e['event']=='ready')['backend'].replace('prefill_alias_groups','prefill_alias_groups_BAD')
        elif label == 'alias-layout':next(e for e in es if e['event']=='native_prefill_alias_prepared')['layout'][0]['slots']+=1
        elif label == 'missing-alias-preparation':es.remove(next(e for e in es if e['event']=='native_prefill_alias_prepared'))
        elif label == 'alias-lifetime':next(e for e in es if e['event']=='ready')['backend']=next(e for e in es if e['event']=='ready')['backend'].replace('prefill_lifetime_reuse','prefill_lifetime_reuse_BAD')
        elif label == 'grouped-value-policy':
            ready=next(e for e in es if e['event']=='ready');ready['backend']=ready['backend'].replace('mistral_grouped_value_context_two_heads','mistral_grouped_value_context_two_heads_BAD')
        elif label == 'pool-policy':next(e for e in es if e['event']=='ready')['backend']=next(e for e in es if e['event']=='ready')['backend'].replace('native_prefill_hidden_pool','native_prefill_hidden_pool_BAD')
        elif label == 'pool-bindings':next(e for e in es if e['event']=='native_prefill_pool' and e['bindings'])['bindings']=0
        elif label == 'missing-pool-route':es.remove(next(e for e in es if e['event']=='native_prefill_pool'))
        else: es[-1]['memory'][OWNERS[0]] = 1
        try:
            validate(es, cs, ct, profile, exit_status)
        except (ValueError, KeyError, IndexError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted corruption: '+label)
    raw(Path(__file__))
    report.update(collection_integrity_passed=True, exact_native_coding_control_cases=len(controls),
                  corruptions_rejected=rejected, files=files, scope=__doc__)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('collection_integrity_passed', 'exact_native_coding_control_cases',
                                          'maximum_active_rows', 'close_allocation_bytes')}))


if __name__ == '__main__':
    main()
