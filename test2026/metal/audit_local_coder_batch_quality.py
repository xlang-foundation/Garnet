"""Audit interleaved native coding streams against frozen inputs after shutdown.

This proves transport/accounting and batch execution; sandbox code evaluation
is separate. Native request IDs, rather than arrival order, join the records.
"""
import argparse,hashlib,json
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def native_events(path):
    for line in path.read_text().splitlines():
        outer=json.loads(line)
        try:event=json.loads(outer['line'])
        except json.JSONDecodeError:continue
        if isinstance(event,dict):yield event

def indexed_decode_issues(profile,calls,expected_decode_calls=None,require_sparse=True):
    if not profile.get('active_decode_projections',False):
        return ['Unexpected active FP32 decode projections in control profile'] if calls else []
    issues=[];coverage={stage:profile.get('expected_'+stage+'_indexed_decode_projections') for stage in ('body','head')}
    if any(type(value) is not int or value<=0 for value in coverage.values()):return ['Invalid frozen active FP32 projection coverage']
    coverage['decode']=coverage['body']+coverage['head']
    if expected_decode_calls is not None and sum(e.get('stage')=='decode' for e in calls)!=expected_decode_calls:
        issues.append('Indexed FP32 decode command count differs from native plans')
    if require_sparse:
        for stage in ('decode','head'):
            if stage=='decode' and expected_decode_calls==0:continue
            if not any(e.get('stage')==stage and type(e.get('affine_dispatches')) is int and e['affine_dispatches']>0 for e in calls):
                issues.append('Required active FP32 '+stage+' projection coverage missing')
    for event in calls:
        stage=event.get('stage');valid=event.get('active_rows');physical=event.get('physical_rows');count=event.get('affine_dispatches')
        if stage not in ('body','head','decode') or any(type(value) is not int for value in (valid,physical,count)):
            issues.append('Active FP32 decode projection route/coverage differs');continue
        expected=coverage[stage] if valid<physical else 0
        if physical!=profile['batch'] or not 0<valid<=physical or count!=expected:
            issues.append('Active FP32 decode projection route/coverage differs')
    return issues

def affine_profile_issues(profile,calls,expected_counts):
    if not profile.get('bounded_affine_kernel_profiles',False):
        return ['Unexpected bounded affine kernel profile events'] if calls else []
    coverage={stage:profile.get('expected_'+stage+'_affine_projections') for stage in ('body','head')}
    if any(type(value) is not int or value<=0 for value in coverage.values()):return ['Invalid frozen bounded affine projection coverage']
    coverage['prefill']=coverage.pop('body');coverage['decode']=coverage['prefill']+coverage['head']
    issues=[]
    if not calls or not any(e.get('stage')=='head' for e in calls):issues.append('Required bounded affine profile head evidence missing')
    for stage,count in expected_counts.items():
        if count is not None and sum(e.get('stage')==stage for e in calls)!=count:
            issues.append('Bounded affine profile '+stage+' command count differs from native plans')
    for event in calls:
        stage=event.get('stage');valid=event.get('active_rows');physical=event.get('physical_rows')
        count=event.get('affine_dispatches');misses=event.get('misses');pipelines=event.get('pipeline_count')
        if stage not in coverage or any(type(value) is not int for value in (valid,physical,count,misses,pipelines)):
            issues.append('Bounded affine profile route/coverage differs');continue
        index_bits=profile.get('affine_profile_index_bits',64)
        if type(index_bits) is not int or index_bits not in (32,64) or event.get('index_bits',64)!=index_bits:
            issues.append('Bounded affine profile index width differs')
        thread_limit=profile.get('affine_profile_max_threads',0)
        if type(thread_limit) is not int or thread_limit not in (0,128) or (thread_limit==128 and event.get('pipeline_max_threads')!=128):
            issues.append('Bounded affine profile compiler thread limit differs')
        expected_rows=profile['batch']*(profile['prefill_chunk'] if stage=='prefill' else 1)
        if (physical!=expected_rows or not 0<valid<=physical or count!=coverage[stage] or misses!=0 or not 0<pipelines<=512):
            issues.append('Bounded affine profile route/coverage differs')
    return issues

def profile_close_issues(profile,memory):
    if not profile.get('bounded_affine_kernel_profiles',False):return []
    limit=profile.get('compiled_engine_close_bytes_limit');allocated=memory.get('metal_allocated_bytes')
    issues=[]
    if type(limit) is not int or not 0<limit<=4*1024*1024 or type(allocated) is not int or not 0<=allocated<=limit:
        issues.append('Compiled engine post-close allocation exceeds its frozen limit')
    for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
        if type(memory.get(key)) is not int or memory[key]!=0:issues.append('Native compiled profile resource owners remain or are unobserved')
    return issues

def chunk_attention_issues(profile,events,backend,prefill_calls,decode_calls):
    issues=[]
    for policy in ('chunk_probability_reuse','chunk_attention_index32'):
        enabled=profile.get(policy,False);calls=[e for e in events if e.get('event')=='native_'+policy]
        if type(enabled) is not bool or enabled!=(policy in backend):issues.append('Chunk attention backend policy differs: '+policy)
        if not enabled:
            if calls:issues.append('Unexpected chunk attention experiment events: '+policy)
            continue
        coverage=profile.get('expected_chunk_attention_dispatches')
        if type(coverage) is not int or coverage<1:issues.append('Invalid frozen chunk attention coverage');continue
        if (sum(e.get('stage')=='prefill' for e in calls)!=prefill_calls or
            sum(e.get('stage')=='decode' for e in calls)!=decode_calls):issues.append('Chunk attention command coverage differs')
        for event in calls:
            stage=event.get('stage');count=event.get('dispatches')
            if stage not in ('prefill','decode') or type(count) is not int or count!=(coverage if stage=='prefill' else 0):
                issues.append('Chunk attention route/count differs')
    if profile.get('chunk_probability_reuse') and profile.get('chunk_attention_index32'):issues.append('Attention experiments must be isolated')
    return issues

def audit(root,reference=None):
    issues=[];artifacts=json.loads((root/'artifacts.json').read_text());collection=json.loads((root/'collection.json').read_text())
    for name,expected in artifacts['files'].items():
        if digest(root/'frozen'/name)!=expected:issues.append('Frozen artifact changed: '+name)
    if digest(root/'libgarnet.dylib')!=artifacts['library_sha256']:issues.append('Native library changed')
    if collection.get('status')!='collected_pending_evaluation' or collection.get('client_exit_code')!=0 or collection['issues']:issues.append('Collection incomplete or failed')
    if collection.get('server_exit_code') not in (0,-2):issues.append('Unexpected bridge exit')
    responses=root/'responses';manifest=json.loads((responses/'frozen/manifest.json').read_text());status=json.loads((responses/'status.json').read_text());plans=manifest['plans']
    if status['status']!='collected_pending_sandbox_evaluation' or status.get('parallel_tasks')!=8:issues.append('Concurrent coding collection differs')
    if status['tasks']!=[dict(id=plan['id'],client_exit_code=0) for plan in plans]:issues.append('Task set/status differs')
    ready=[];closed=[];completed={};streams={};batch_plans=[];prompt_counts={};mpp=[]
    chunks={};decode_events=[];mixed_events=[];indexed_batches=[];indexed_events=[];reuse64_events=[];indexed_decode_events=[];affine_profile_events=[]
    events=list(native_events(root/'server/worker-stdout.jsonl'))
    for event_index,event in enumerate(events):
        kind=event.get('event');key=event.get('id')
        if kind=='ready':ready.append(event)
        elif kind=='closed':closed.append(event)
        elif kind=='native_batch':
            batch_plans.append(event)
            indexed_batches.append((event_index,event))
            if event['stage']=='decode':decode_events.append((event_index,event))
        elif kind=='native_prefill_chunk':chunks.setdefault(event['native_request'],[]).append((event_index,event))
        elif kind=='native_mpp':mpp.append(event)
        elif kind=='native_indexed':indexed_events.append(event)
        elif kind=='native_reuse64':reuse64_events.append(event)
        elif kind=='native_indexed_decode':indexed_decode_events.append(event)
        elif kind=='native_affine_profile':affine_profile_events.append(event)
        elif kind=='native_mixed_forward':mixed_events.append((event_index,event))
        elif kind=='error':issues.append('Native request/runtime error: '+event.get('error',''))
        elif kind=='token':
            tokens=streams.setdefault(key,[]);tokens.append(event['token_id'])
            if key in completed or event['sampled_tokens']!=len(tokens) or event.get('cached_tokens')!=0:issues.append('Native token sequence/cache usage differs')
            if prompt_counts.setdefault(key,event['prompt_tokens'])!=event['prompt_tokens']:issues.append('Native prompt count changed')
        elif kind=='completed':
            if key in completed:issues.append('Native completion identity reused')
            completed[key]=event
            if event['generated_ids']!=streams.get(key) or event.get('cached_tokens')!=0:issues.append('Native streamed/final IDs or cache usage differ')
    if len(ready)!=1 or ready[0].get('active_requests')!=2 or ready[0].get('prefix_cache') is not False:issues.append('Ready profile differs from uncached batch2')
    profiles=[]
    for name in artifacts['files']:
        if name.endswith('.json'):
            candidate=json.loads((root/'frozen'/name).read_text())
            if isinstance(candidate,dict) and 'profile_id' in candidate:profiles.append(candidate)
    if len(profiles)!=1:raise ValueError('Exactly one hash-verified frozen native profile required')
    profile=profiles[0]
    issues.extend(chunk_attention_issues(profile,events,ready[0].get('backend','') if ready else '',
        sum(e['stage']=='prefill' for e in batch_plans),len(decode_events)-len(mixed_events)))
    if profile.get('require_reuse64',False):
        if not reuse64_events or not any(e.get('affine_dispatches',0)>0 for e in reuse64_events):
            issues.append('Required cooperative64 projection route missing')
        for event in reuse64_events:
            count=event.get('affine_dispatches',-1);logical=event.get('logical_rows',-1);physical=event.get('physical_rows',-1)
            if (physical!=profile['batch']*profile['prefill_chunk'] or not 0<logical<=physical or
                count<0 or (logical>=64 and count<1) or (logical<64 and count!=0)):
                issues.append('Cooperative64 projection route/count differs')
    elif reuse64_events:issues.append('Unexpected cooperative64 projection in control profile')
    issues.extend(indexed_decode_issues(profile,indexed_decode_events,len(decode_events)-len(mixed_events)))
    issues.extend(affine_profile_issues(profile,affine_profile_events,dict(
        prefill=sum(e['stage']=='prefill' for e in batch_plans) if profile.get('prefill_schedule')=='chunked' else None,
        decode=len(decode_events)-len(mixed_events),
        head=sum(e.get('stage')=='head' for e in indexed_decode_events) if profile.get('active_decode_projections') else None)))
    if ready and profile.get('bounded_affine_kernel_profiles',False)!=('bounded_affine_kernel_profiles' in ready[0].get('backend','')):
        issues.append('Bounded kernel compilation policy differs from native backend')
    if profile.get('indexed_projections',False):
        if not indexed_events or not any(e.get('affine_dispatches',0)>0 for e in indexed_events):
            issues.append('Required indexed projection route missing')
        for event in indexed_events:
            count=event.get('affine_dispatches',-1);valid=event.get('active_tokens',-1)
            physical=event.get('physical_rows',-1)
            if (physical!=profile['batch']*profile['prefill_chunk'] or not 0<valid<=physical or
                count<0 or (valid<physical and count<1) or (valid==physical and count!=0)):
                issues.append('Indexed projection sparse route/count differs')
    elif indexed_events:issues.append('Unexpected indexed projections in control profile')
    if profile.get('mixed_forward',False):
        if not mixed_events:issues.append('Required mixed forward missing')
        previous=-1
        for index,event in mixed_events:
            preceding=[plan for i,plan in indexed_batches if previous<i<index][-2:]
            expected=[];offset=0
            for stage in ('decode','prefill'):
                for plan in preceding:
                    if plan['stage']!=stage:continue
                    for request in plan['requests']:
                        count=1 if stage=='decode' else request['prefill_tokens']
                        expected.append(dict(id=request['id'],row=request['row'],epoch=request['epoch'],stage=stage,
                            packed_offset=offset,token_count=count,slot=request['slot'],
                            absolute_position=request['slot']+request['position_delta']))
                        offset+=count
            if ({plan['stage'] for plan in preceding}!={'decode','prefill'} or
                event.get('segments')!=expected or event.get('actual_tokens')!=offset or
                event.get('projection_rows')!=profile['batch']*profile['prefill_chunk'] or
                len({segment['row'] for segment in expected})!=len(expected)):
                issues.append('Mixed forward differs from native tick')
            previous=index
    elif mixed_events:issues.append('Unexpected mixed forward in control profile')
    if profile.get('require_mpp',False) and (not mpp or any(event.get('affine_dispatches',0)<1 for event in mpp)):issues.append('Required native MPP route missing')
    if len(ready)==1 and (ready[0].get('profile')!=profile or ready[0].get('capacity')!=profile['capacity']):issues.append('Frozen context/profile differs from ready event')
    if len(ready)==1 and ready[0]['memory']['metal_allocated_bytes']>=ready[0]['memory']['metal_recommended_working_set_bytes']:issues.append('Ready device working set exceeded')
    if len(closed)!=1:issues.append('Native close allocation differs')
    elif profile.get('bounded_affine_kernel_profiles',False):issues.extend(profile_close_issues(profile,closed[0]['memory']))
    elif closed[0]['memory']['metal_allocated_bytes']!=720896:issues.append('Native close allocation differs')
    exit_record=json.loads((root/'server/worker-exit.json').read_text())
    if exit_record.get('returncode')!=0 or exit_record.get('expected_close') is not True:issues.append('Native worker did not exit cleanly')
    if len(completed)!=len(plans) or set(completed)!=set(streams):issues.append('Native completion count/set differs')
    reference_by_prompt={}
    if reference is not None:
        for event in native_events(reference):
            if event.get('event')=='completed':reference_by_prompt[tuple(event['prompt_ids'])]=event['generated_ids']
    results=[];matched=set()
    for plan in plans:
        name=plan['id'];frozen=responses/'frozen'/name
        cases=json.loads((frozen/'cases.json').read_text());workload=json.loads((frozen/'workload.json').read_text())
        row=json.loads((responses/name/'request-0.result.json').read_text())
        expected=workload['requests'][0]['token_ids']
        if len(cases)!=1 or len(workload['requests'])!=1 or digest(frozen/'cases.json')!=plan['cases_sha256'] or workload['cases_sha256']!=plan['cases_sha256']:issues.append(name+': frozen input changed')
        http=[];done=0
        for line in (responses/name/'request-0.events.jsonl').read_text().splitlines():
            data=json.loads(line)['data']
            if data=='[DONE]':done+=1
            else:http.append(json.loads(data))
        if not http:issues.append(name+': empty HTTP stream');continue
        key=http[0]['id'];last=http[-1];native=completed.get(key)
        if native is None:issues.append(name+': missing native request');continue
        if key in matched:issues.append(name+': duplicated HTTP/native request mapping')
        matched.add(key)
        ids=[event['token_id'] for event in http if 'token_id' in event]
        if done!=1 or any(event.get('id')!=key for event in http) or ids!=native['generated_ids'] or last.get('response_token_ids')!=ids:issues.append(name+': HTTP/native output IDs differ')
        if last.get('prompt_token_ids')!=expected or native['prompt_ids']!=expected:issues.append(name+': frozen prompt IDs differ')
        if any(event.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens')!=0 for event in http):issues.append(name+': HTTP cache policy differs')
        if not 1<=len(ids)<=cases[0]['max_tokens'] or any(type(token) is not int or token<0 for token in ids):issues.append(name+': invalid output IDs/budget')
        if row.get('status')!='completed' or row.get('cached_prompt_tokens')!=0 or row.get('prompt_tokens')!=len(expected) or row.get('generated_tokens')!=len(ids):issues.append(name+': client accounting differs')
        if row.get('text')!=native['text'] or row.get('finish_reason')!=native['finish_reason']:issues.append(name+': text/finish reason differs')
        if last['usage']['prompt_tokens']!=len(expected) or last['usage']['completion_tokens']!=len(ids):issues.append(name+': final SSE usage differs')
        if native['memory']['metal_allocated_bytes']>=native['memory']['metal_recommended_working_set_bytes']:issues.append(name+': device working set exceeded')
        entry=dict(id=name,prompt_tokens=len(expected),generated_tokens=len(ids),native_row=native['native_row'],native_epoch=native['native_epoch'],generated_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
        if reference is not None:entry['matches_single_reference']=reference_by_prompt.get(tuple(expected))==ids
        results.append(entry)
    if matched!=set(completed):issues.append('Unmatched native completions')
    combined=sum(plan['stage']=='decode' and sum(plan['active_rows'])==2 for plan in batch_plans)
    if combined<1:issues.append('No actual combined decode plan')
    interleavings=0
    if profile.get('prefill_schedule','whole')=='chunked':
        if set(chunks)!={event['native_request'] for event in completed.values()}:issues.append('Chunked prefill request set differs')
        for event in completed.values():
            parts=chunks.get(event['native_request'],[]);offset=0
            for index,(_,part) in enumerate(parts):
                count=min(profile['prefill_chunk'],len(event['prompt_ids'])-offset)
                if (part['offset']!=offset or part['tokens']!=count or count<=0 or
                    part['row']!=event['native_row'] or part['epoch']!=event['native_epoch'] or
                    part['final']!=(index==len(parts)-1)):issues.append('Native prefill chunk progression/lease differs')
                offset+=part['tokens']
            if offset!=len(event['prompt_ids']):issues.append('Native prefill chunks do not cover prompt')
            for (left,_),(right,_) in zip(parts,parts[1:]):
                interleavings+=sum(left<i<right and any(r['id']!=event['native_request'] for r in plan['requests']) for i,plan in decode_events)
        if interleavings<1:issues.append('No existing decode advanced between native prefill chunks')
    elif chunks:issues.append('Whole-prompt profile emitted chunk scheduling evidence')
    epochs=[event['native_epoch'] for event in completed.values()]
    if len(set(epochs))!=len(plans):issues.append('Request epochs were reused')
    if set(event['native_row'] for event in completed.values())!={0,1}:issues.append('Physical row profile differs')
    return dict(status='passed' if not issues else 'failed',issues=issues,tasks=results,combined_decode_plans=combined,prefill_decode_interleavings=interleavings,
        context_capacity=ready[0]['capacity'] if ready else None,mpp_prefill_calls=len(mpp),mixed_forward_calls=len(mixed_events),
        indexed_prefill_calls=len(indexed_events),
        reuse64_prefill_calls=len(reuse64_events),indexed_decode_calls=len(indexed_decode_events),affine_profile_calls=len(affine_profile_events),affine_profile_index_bits=profile.get('affine_profile_index_bits',64) if profile.get('bounded_affine_kernel_profiles') else 0,
        chunk_attention_index32_calls=sum(e.get('event')=='native_chunk_attention_index32' for e in events),
        scope='Concurrent native frozen prompt/stream/HTTP accounting and row lifecycle; coding correctness requires separate sandbox evaluation',
        auditor_sha256=digest(Path(__file__)),reference_worker_log_sha256=digest(reference) if reference else None,
        artifacts_sha256=digest(root/'artifacts.json'),worker_log_sha256=digest(root/'server/worker-stdout.jsonl'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--reference-worker-log',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=audit(args.root,args.reference_worker_log)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],issues=report['issues'],tasks=len(report['tasks']),combined_decode_plans=report['combined_decode_plans'])))
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
