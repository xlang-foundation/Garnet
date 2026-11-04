"""Join frozen serving workloads, raw SSE and native batched token streams.

Timing admission evidence is distinct from repeated performance qualification.
"""
import argparse,hashlib,json,math
from pathlib import Path
from summarize_local_coder_baseline import audit as baseline_audit
from audit_local_coder_batch_quality import native_events

def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(root):
    campaign=root/'campaign';base=baseline_audit(campaign);issues=list(base['issues'])
    spec=read(root/'engine.json');metadata=read(campaign/'frozen/metadata.json')
    if metadata['engine']!=spec:issues.append('Engine specification differs from campaign freeze')
    for name,wanted in spec['sources'].items():
        if digest(root/'sources'/name)!=wanted:issues.append('Frozen source/library differs: '+name)
    profile=read(root/'sources/profile.json');manifest=metadata['manifest']
    events=list(native_events(root/'server/worker-stdout.jsonl'))
    ready=[e for e in events if e.get('event')=='ready'];closed=[e for e in events if e.get('event')=='closed']
    if len(ready)!=1 or ready[0].get('profile')!=profile:issues.append('Native ready profile differs')
    if profile['prefix_cache'] is not False:issues.append('Matched uncached policy required')
    if any(e.get('event')=='error' for e in events):issues.append('Native error event present')
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
    for folder in sorted(campaign.glob('cohort-*')):
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
        exit_path=root/'server/worker-exit.json'
        exit_record=read(exit_path) if exit_path.exists() else {}
        if len(closed)!=1 or exit_record.get('returncode')!=0 or not exit_record.get('expected_close'):issues.append('Requested graceful native shutdown evidence missing')
    if not closed:shutdown['limitations'].append('No graceful native close snapshot; rely on separate coding cleanup gate and collector group termination evidence')
    if any(e.get('memory',{}).get('metal_allocated_bytes',0)>1024*1024 for e in closed):issues.append('Observed native cleanup exceeds baseline')
    return dict(status='passed' if not issues else 'failed',issues=issues,scope=manifest['scope'],requests=requests,cells=cells,
        combined_decode_plans=combined,combined_decode_required=combined_required,
        shutdown=shutdown,baseline_accounting=base,worker_log_sha256=digest(root/'server/worker-stdout.jsonl'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();report=audit(args.root)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],issues=report['issues'],requests=len(report['requests']),combined_decode_plans=report['combined_decode_plans'],shutdown=report['shutdown'])))
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
