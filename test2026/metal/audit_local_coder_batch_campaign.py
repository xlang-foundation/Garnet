"""Join frozen serving workloads, raw SSE and native batched token streams.

Timing admission evidence is distinct from repeated performance qualification.
"""
import argparse,hashlib,json,math
from pathlib import Path
from summarize_local_coder_baseline import audit as baseline_audit
from audit_local_coder_batch_quality import native_events,indexed_decode_issues,affine_profile_issues,profile_close_issues,chunk_attention_issues,prefill_alias_issues,linear_chunk_active_issues

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(root,cohort_number=None,_base=None):
    campaign=root/'campaign';base=_base if _base is not None else baseline_audit(campaign);issues=list(base['issues'])
    spec=read(root/'engine.json');metadata=read(campaign/'frozen/metadata.json')
    if metadata['engine']!=spec:issues.append('Engine specification differs from campaign freeze')
    for name,wanted in spec['sources'].items():
        if digest(root/'sources'/name)!=wanted:issues.append('Frozen source/library differs: '+name)
    if 'benchmark-manifest.json' in spec['sources'] and metadata['manifest']!=read(root/'sources/benchmark-manifest.json'):
        issues.append('Frozen native benchmark manifest differs')
    manifest=metadata['manifest'];required=manifest.get('required_fresh_process_cohorts',manifest.get('cohorts',1))
    if cohort_number is None and required>1:
        parts=[];identities=[];starts=[]
        keys={str(n) for n in range(1,required+1)}
        if set(spec.get('native_cohort_profiles',{}))!=keys or set(spec.get('native_cohort_log_roots',{}))!=keys:
            issues.append('Native cohort profile/log coverage differs')
        canonical=read(root/'sources/profile.json');canonical.pop('memory_inventory_path',None)
        for number in range(1,required+1):
            key=str(number);folder=campaign/f'cohort-{number}'
            log_root=root/spec.get('native_cohort_log_roots',{}).get(key,f'server/cohort-{number}')
            profile_path=root/'sources'/spec.get('native_cohort_profiles',{}).get(key,f'profile-{number}.json')
            if not folder.exists() or not (log_root/'worker-stdout.jsonl').exists() or not profile_path.exists():
                issues.append('Missing native cohort evidence: '+key);continue
            profile=read(profile_path);profile.pop('memory_inventory_path',None)
            if profile!=canonical:issues.append('Native cohort runtime profile differs')
            state=read(folder/'cohort.json');identities.append((state.get('pid'),state.get('spawn_monotonic')));starts.append(state.get('spawn_monotonic'))
            expected=[arg.replace('{cohort}',key) for arg in spec['argv']]
            if state.get('server_argv')!=expected:issues.append('Resolved native cohort argv differs')
            part=audit(root,number,base);parts.append(dict(number=number,report=part));issues.extend(part['issues'])
        if len(identities)!=required or len(set(identities))!=required or any(type(t) not in (int,float) or not math.isfinite(t) for t in starts) or any(a>=b for a,b in zip(starts,starts[1:])):
            issues.append('Native cohorts do not have distinct fresh-process identities')
        issues=list(dict.fromkeys(issues));reports=[p['report'] for p in parts]
        totals={name:sum(p.get(name,0) for p in reports) for name in ('combined_decode_plans','affine_profile_calls','indexed_decode_calls','indexed_decode_command_calls','chunk_attention_index32_calls','chunk_attention_mps_calls','chunk_attention_bucket_calls')}
        return dict(status='passed' if not issues else 'failed',issues=issues,scope=manifest['scope'],cohorts=parts,
            requests=[dict(r,cohort=p['number']) for p in parts for r in p['report']['requests']],
            cells=[dict(c,cohort=p['number']) for p in parts for c in p['report']['cells']],**totals,
            combined_decode_required=any(p.get('combined_decode_required') for p in reports),
            prefill_alias_groups=all(p.get('prefill_alias_groups',False) for p in reports) if reports else False,
            linear_chunk_active_only=all(p.get('linear_chunk_active_only',False) for p in reports) if reports else False,
            indexed_decode_sparse_required=any(p.get('indexed_decode_sparse_required') for p in reports),
            shutdown=dict(mode='Sequential fresh native cohorts',native_close_observed=len(reports)==required and all(p['shutdown']['native_close_observed'] for p in reports),cohorts=[p['shutdown'] for p in reports]),
            baseline_accounting=base,auditor_sha256=digest(Path(__file__)),decode_auditor_sha256=digest(Path(__file__).with_name('audit_local_coder_batch_quality.py')))
    profile_name=spec.get('native_cohort_profiles',{}).get(str(cohort_number),'profile.json')
    profile=read(root/'sources'/profile_name)
    server=root/spec.get('native_cohort_log_roots',{}).get(str(cohort_number),'server')
    events=list(native_events(server/'worker-stdout.jsonl'))
    ready=[e for e in events if e.get('event')=='ready'];closed=[e for e in events if e.get('event')=='closed']
    if len(ready)!=1 or ready[0].get('profile')!=profile:issues.append('Native ready profile differs')
    if profile['prefix_cache'] is not False:issues.append('Matched uncached policy required')
    issues.extend(prefill_alias_issues(profile,events,ready[0].get('backend','') if ready else ''))
    issues.extend(linear_chunk_active_issues(profile,ready[0].get('backend','') if ready else ''))
    if any(e.get('event')=='error' for e in events):issues.append('Native error event present')
    indexed_decode=[e for e in events if e.get('event')=='native_indexed_decode']
    pure_decodes=sum(e.get('event')=='native_batch' and e.get('stage')=='decode' for e in events)-sum(e.get('event')=='native_mixed_forward' for e in events)
    issues.extend(chunk_attention_issues(profile,events,ready[0].get('backend','') if ready else '',
        sum(e.get('event')=='native_batch' and e.get('stage')=='prefill' for e in events),pure_decodes))
    require_sparse=profile['batch']>1 and any(len(read(campaign/'frozen'/plan['id']/'cases.json'))==1 for plan in manifest['plans'])
    issues.extend(indexed_decode_issues(profile,indexed_decode,pure_decodes,require_sparse))
    affine_profiles=[e for e in events if e.get('event')=='native_affine_profile']
    issues.extend(affine_profile_issues(profile,affine_profiles,dict(
        prefill=sum(e.get('event')=='native_batch' and e.get('stage')=='prefill' for e in events) if profile.get('prefill_schedule')=='chunked' else None,
        decode=pure_decodes,head=sum(e.get('stage')=='head' for e in indexed_decode) if profile.get('active_decode_projections') else None)))
    if ready and profile.get('bounded_affine_kernel_profiles',False)!=('bounded_affine_kernel_profiles' in ready[0].get('backend','')):
        issues.append('Bounded kernel compilation policy differs from native backend')
    streams={};completed={};batches=[]
    for e in events:
        kind=e.get('event');key=e.get('id')
        if kind=='native_batch':
            batches.append(e)
            if len(e['active_rows'])!=profile['batch'] or any(v not in (0,1) for v in e['active_rows']) or sum(e['active_rows'])!=len(e['requests']):issues.append('Native batch row plan differs from frozen capacity')
        elif kind=='token':
            ids=streams.setdefault(key,[]);ids.append(e['token_id'])
            if e['sampled_tokens']!=len(ids) or e.get('cached_tokens')!=0:issues.append('Native streaming usage/cache mismatch')
        elif kind=='completed':
            if key in completed:issues.append('Duplicate native completion')
            completed[key]=e
            if e['generated_ids']!=streams.get(key) or e.get('cached_tokens')!=0:issues.append('Native final output/cache mismatch')
    if set(completed)!=set(streams):issues.append('Native stream/completion sets differ')
    matched=set();requests=[];cells=[];combined_required=False
    folders=[campaign/f'cohort-{cohort_number}'] if cohort_number is not None else sorted(campaign.glob('cohort-*'))
    for folder in folders:
        for plan in manifest['plans']:
            frozen=campaign/'frozen'/plan['id'];workload=read(frozen/'workload.json');bodies=read(frozen/'cases.json')
            if profile['batch']>1 and len(bodies)>1:combined_required=True
            for phase,count in [('warmup',manifest['warmup_waves_per_cell']),('measured',manifest['measured_waves_per_cell'])]:
                for wave in range(count):
                    path=folder/f"{plan['id']}-{phase}-{wave}";summary=read(path/'summary.json');local=[]
                    for index,body in enumerate(bodies):
                        result=read(path/f'request-{index}.result.json');expected=workload['requests'][index]['token_ids'];http=[];done=0
                        for line in (path/f'request-{index}.events.jsonl').read_text().splitlines():
                            payload=json.loads(line)['data']
                            if payload=='[DONE]':done+=1
                            else:http.append(json.loads(payload))
                        if not http:issues.append('Empty SSE stream');continue
                        key=http[0]['id'];last=http[-1];native=completed.get(key)
                        if native is None:issues.append('SSE request has no native completion');continue
                        if key in matched:issues.append('Native completion matched more than once')
                        matched.add(key);ids=[e['token_id'] for e in http if 'token_id' in e]
                        if done!=1 or any(e.get('id')!=key for e in http) or ids!=native['generated_ids'] or last.get('response_token_ids')!=ids:issues.append('SSE/native token IDs differ')
                        if last.get('prompt_token_ids')!=expected or native['prompt_ids']!=expected:issues.append('Frozen prompt token IDs differ')
                        if any(e.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens')!=0 for e in http):issues.append('SSE cache policy differs')
                        if result.get('text')!=native['text'] or result.get('finish_reason')!=native['finish_reason']:issues.append('Client/native text or finish differs')
                        if result.get('generated_tokens')!=len(ids) or last['usage']['completion_tokens']!=len(ids) or result.get('prompt_tokens')!=len(expected) or last['usage']['prompt_tokens']!=len(expected):issues.append('Client/native token counts differ')
                        if not 0<len(ids)<=body['max_tokens']:issues.append('Output budget differs')
                        if result['finish_reason']=='length' and len(ids)!=body['max_tokens']:issues.append('Length finish did not consume output budget')
                        for field in ('ttft_content_seconds','decode_tokens_per_second','end_to_end_seconds'):
                            if not math.isfinite(result[field]) or result[field]<=0:issues.append('Invalid timing sample')
                        memories=[native['memory']]+([ready[0]['memory']] if len(ready)==1 else [])
                        for memory in memories:
                            if memory['metal_allocated_bytes']>memory['metal_recommended_working_set_bytes']-256*1024*1024:issues.append('Sampled native workspace exceeds guard')
                        if native['native_row'] not in range(profile['batch']) or native['native_epoch']<=0:issues.append('Invalid native row/epoch')
                        requests.append(dict(id=key,cell=plan['id'],phase=phase,wave=wave,prompt_tokens=len(expected),output_tokens=len(ids),native_row=native['native_row'],native_epoch=native['native_epoch']))
                        local.append(result)
                    tps=None
                    if len(local)==len(bodies) and all(r['status']=='completed' for r in local):
                        duration=max(r['completed'] for r in local)-min(r['submitted'] for r in local)
                        tps=sum(r['generated_tokens'] for r in local)/duration if duration>0 else None
                        recorded=summary.get('aggregate_output_tokens_per_second')
                        if tps is None or recorded is None or not math.isclose(recorded,tps,rel_tol=1e-9):issues.append('Aggregate throughput arithmetic differs')
                    else:issues.append('Wave lacks complete matched native/client results')
                    cells.append(dict(cell=plan['id'],phase=phase,wave=wave,matched_requests=len(local),aggregate_output_tokens_per_second=tps))
    if matched!=set(completed):issues.append('Native completions unmatched to client results')
    if len({r['native_epoch'] for r in requests})!=len(requests):issues.append('Native request epochs reused')
    combined=sum(e['stage']=='decode' and sum(e['active_rows'])>1 for e in batches)
    if combined_required and combined<1:issues.append('No actual combined decode')
    # The legacy timing collector stops the owned process group via SIGTERM.
    # Record this explicitly rather than asserting a graceful native close.
    graceful=spec.get('graceful_native_shutdown',False)
    shutdown=dict(mode='root SIGINT then owned group fallback' if graceful else 'owned process-group SIGTERM',native_close_observed=bool(closed),limitations=[])
    if graceful:
        exit_path=server/'worker-exit.json'
        exit_record=read(exit_path) if exit_path.exists() else {}
        if len(closed)!=1 or exit_record.get('returncode')!=0 or not exit_record.get('expected_close'):issues.append('Requested graceful native shutdown evidence missing')
    if not closed:shutdown['limitations'].append('No graceful native close snapshot; rely on separate coding cleanup gate and collector group termination evidence')
    if profile.get('bounded_affine_kernel_profiles',False):
        if len(closed)!=1:issues.append('Compiled engine native cleanup evidence missing')
        for event in closed:issues.extend(profile_close_issues(profile,event.get('memory',{})))
    elif any(e.get('memory',{}).get('metal_allocated_bytes',0)>1024*1024 for e in closed):issues.append('Observed native cleanup exceeds baseline')
    return dict(status='passed' if not issues else 'failed',issues=issues,scope=manifest['scope'],requests=requests,cells=cells,
        combined_decode_plans=combined,combined_decode_required=combined_required,
        shutdown=shutdown,affine_profile_calls=len(affine_profiles),indexed_decode_calls=len(indexed_decode),indexed_decode_command_calls=sum(e.get('stage')=='decode' for e in indexed_decode),
        chunk_attention_index32_calls=sum(e.get('event')=='native_chunk_attention_index32' for e in events),
        chunk_attention_mps_calls=sum(e.get('event')=='native_chunk_attention_mps' for e in events),
        chunk_attention_bucket_calls=sum(e.get('event')=='native_chunk_attention_bucket' for e in events),
        prefill_alias_groups=profile.get('prefill_alias_groups',False),
        linear_chunk_active_only=profile.get('linear_chunk_active_only',False),
        indexed_decode_sparse_required=require_sparse,baseline_accounting=base,worker_log_sha256=digest(server/'worker-stdout.jsonl'),
        auditor_sha256=digest(Path(__file__)),decode_auditor_sha256=digest(Path(__file__).with_name('audit_local_coder_batch_quality.py')))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();report=audit(args.root)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],issues=report['issues'],requests=len(report['requests']),combined_decode_plans=report['combined_decode_plans'],shutdown=report['shutdown'])))
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
