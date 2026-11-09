"""Audit saved C1/C4 native waves and reconstruct frozen reference metrics.

Read-only inference artifacts; no models are executed by this auditor.
"""
from pathlib import Path
import copy, hashlib, json, math, statistics, sys

ROOT = Path(sys.argv[1]).resolve()
PACKAGE = Path(__file__).resolve().parent
M = ROOT.parent
sys.path.insert(0, str(PACKAGE / 'audit-tools1'))
from audit_half_transport1 import require, load_client, validate
from audit_policies1 import split_policy, profile32_policy, value_policy, paired_policy, silu_policy, timing

files = {}
def raw(p):
    data = p.read_bytes()
    files[str(p)] = hashlib.sha256(data).hexdigest()
    return data

def read(p):
    return json.loads(raw(p))

def check(events, clients, controls, profile, count):
    require(len(clients) == count * 3 and all(len(c['sse_ids']) == 128 and c['result']['cached_prompt_tokens'] == 0 for c in clients), 'wrong response/cache counts')
    require(profile['half_attention_policy'] is True and profile['prefix_capture_policy'] == 'prefill_only_v1', 'wrong native producer policy')
    require(profile['trace_stage_latency'] is False and profile['prefix_cache'] is False and not any(e['event'] in ['native_stage_trace','native_request_state_witness','native_survivor_restore_witness'] for e in events), 'instrumented or cached serving')
    policies = dict(split_score=split_policy(events,profile),profile32=profile32_policy(events,profile),value=value_policy(events,profile),paired=paired_policy(events,profile),silu=silu_policy(events,profile))
    report = validate(events,clients,controls,profile,dict(returncode=0,expected_close=True),expected_requests=count*3,control_prefixes=False,maximum_chunk=128,prefix_policy=False,required_active_rows=count)
    require(all(x['exact_native_coding_control'] for x in report['requests']), 'missing full saved native control')
    metrics = {label:timing([c for c in clients if c['round']==label]) for label in ['cold','warm1','warm2']}
    return dict(native=report,policies=policies,metrics=metrics,half_dispatches=sum(e['dispatches'] for e in events if e['event']=='native_chunk_half_attention'))

def reference_wave(folder, bodies, prompt_ids):
    clients=[]
    for i, body in enumerate(bodies):
        expected_prompt=len(prompt_ids[i])
        prefix=folder/('request-'+str(i))
        require(read(prefix.with_suffix('.request.json'))==body, 'reference request differs')
        d=read(prefix.with_suffix('.result.json'))
        require(d['status']=='completed' and d['generated_tokens']==128 and d['prompt_tokens']==expected_prompt and d['cached_prompt_tokens']==0, 'invalid frozen reference result')
        require(d['request_sha256']==hashlib.sha256(raw(prefix.with_suffix('.request.json'))).hexdigest(), 'reference request hash mismatch')
        wire=[json.loads(s) for s in raw(prefix.with_suffix('.events.jsonl')).splitlines()]
        require(wire[-1]['data']=='[DONE]' and sum(w['data']=='[DONE]' for w in wire)==1, 'invalid frozen reference stream')
        data=[(w['monotonic'],json.loads(w['data'])) for w in wire if w['data']!='[DONE]']
        require(all(b['monotonic']>=a['monotonic'] for a,b in zip(wire,wire[1:])), 'nonmonotonic SSE')
        content=[at for at,e in data if any(c.get('delta',{}).get('content') for c in e.get('choices',[]))]
        require(content[0]==d['first_content'] and content[-1]==d['last_content'], 'reference content clocks mismatch')
        text=''.join(c.get('delta',{}).get('content','') for _,e in data for c in e.get('choices',[]))
        require(text==d['text'], 'reference content mismatch')
        count=0;advances=[]
        for at,e in data:
            if not e.get('usage'): continue
            u=e['usage'];n=u['completion_tokens']
            require(u['prompt_tokens']==expected_prompt and type(n) is int and count<=n<=128 and (u.get('prompt_tokens_details') or {}).get('cached_tokens',0)==0, 'invalid reference usage')
            if n>count:advances.append((at,n));count=n
        require(count==128 and advances[0][1]==d['first_usage_batch_tokens']==1, 'invalid reference token interval')
        require(advances[0][0]==d['first_usage_arrival'] and advances[-1][0]==d['last_usage_arrival'], 'reference usage clocks mismatch')
        require(all(math.isfinite(d[k]) for k in ['submitted','completed','first_content','last_content','first_usage_arrival','last_usage_arrival','ttft_content_seconds','decode_tokens_per_second']), 'nonfinite reference timing')
        require(d['submitted']<=d['first_content']<=d['last_content']<=d['completed'], 'reference timing order')
        require(math.isclose(d['ttft_content_seconds'],d['first_content']-d['submitted'],rel_tol=2e-12,abs_tol=1e-9), 'false reference TTFT')
        require(math.isclose(d['decode_tokens_per_second'],127/(d['last_usage_arrival']-d['first_usage_arrival']),rel_tol=2e-12), 'false reference decode')
        clients.append(d)
    metrics=dict(ttft_median_seconds=statistics.median(c['ttft_content_seconds'] for c in clients),decode_median_tps=statistics.median(c['decode_tokens_per_second'] for c in clients),aggregate_tps=128*len(clients)/(max(c['completed'] for c in clients)-min(c['submitted'] for c in clients)))
    summary=read(folder/'summary.json')
    require(summary['status']=='completed' and summary['successful_requests']==len(bodies) and math.isclose(summary['aggregate_output_tokens_per_second'],metrics['aggregate_tps'],rel_tol=2e-12), 'frozen reference summary mismatch')
    return metrics

def summarize(rows):
    return dict(measured_waves=len(rows),median={k:statistics.median(x[k] for x in rows) for k in rows[0]},range={k:[min(x[k] for x in rows),max(x[k] for x in rows)] for k in rows[0]})

def main():
    state=read(ROOT/'collection-status1.json');plan=read(ROOT/'plan1.json')
    require(state['status']=='collected_pending_independent_audit' and len(state['waves'])==4 and all(x['status']=='collected' and x['returncode']==0 for x in state['waves']), 'unfinished campaign')
    require([x['label'] for x in state['waves']]==[x['label'] for x in plan['waves']], 'changed collection order')
    require(all(b['started_ns']>=a['finished_ns'] for a,b in zip(state['waves'],state['waves'][1:])), 'overlapping engines')
    require(read(ROOT/'postflight1.json')['passed'], 'missing terminal postflight')
    pre=read(ROOT/'preflight1.json')
    for path,digest in pre['files_sha256'].items():
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        require(h.hexdigest()==digest, 'changed preflight input '+path)
    controls_raw=[json.loads(json.loads(s)['line']) for s in raw(M/'mistral-profile-memory1/admission1/server-1/worker-stdout.jsonl').splitlines()]
    plans={p['id']:p for p in read(ROOT/'serving-manifest1.json')['plans']}
    controls={};prompts={};bodies={}
    for case,p in plans.items():
        workload=read(Path(p['workload']));bodies[case]=read(Path(p['cases']))
        prompts[case]=[x['token_ids'] for x in workload['requests']]
        require(workload['cases_sha256']==p['cases_sha256'] and hashlib.sha256(raw(Path(p['cases']))).hexdigest()==p['cases_sha256'], 'changed frozen workload')
        for i,prompt in enumerate(prompts[case]):
            matches=[e for e in controls_raw if e['event']=='completed' and e['prompt_ids']==prompt and len(e['generated_ids'])==128]
            require(matches and all(e['generated_ids']==matches[0]['generated_ids'] and e['text']==matches[0]['text'] for e in matches), 'missing/inconsistent full FP32 control')
            controls[case+'-'+str(i)]=dict(status='completed',prompt_ids=prompt,generated_ids=matches[0]['generated_ids'],text=matches[0]['text'])
    results=[];datasets=[]
    for wave in plan['waves']:
        folder=ROOT/wave['label'];campaign=folder/'campaign';status=read(campaign/'status.json');phase=status['phases']['observed']
        require(status['status']=='collected_pending_independent_audit' and phase['status']=='collected' and phase['shutdown']['graceful_native_closed'] and not phase['shutdown']['forced_kill'], 'unclean native engine')
        rounds=status['rounds'];require([x['round'] for x in rounds]==['cold','warm1','warm2'] and all(x['status']=='collected' and x['finished_ns']>x['started_ns'] for x in rounds), 'round coverage')
        require(all(b['started_ns']>=a['finished_ns'] for a,b in zip(rounds,rounds[1:])), 'overlapping rounds')
        folder=campaign/'observed';profile=read(folder/'profile.json');manifest=read(campaign/'manifest.json');case=wave['case'];count=len(bodies[case]);clients=[]
        expected_profile=read(ROOT/wave['label']/'profile.json');expected_profile.update(profile_id=expected_profile['profile_id']+'-fresh',memory_inventory_path=str(folder/'prepared-inventory.json'))
        require(profile==expected_profile, 'changed executed profile')
        for i,entry in enumerate(manifest['requests']):
            row=i%count;require(entry['body']==bodies[case][row] and entry['case_id']==case+'-'+str(row), 'changed planned request')
            prefix=folder/entry['label']/'request-0'
            c=load_client(prefix,entry['case_id'],entry['body'],prompts[case][row],read,raw);c['round']=entry['round'];c['wire']=[json.loads(s) for s in raw(prefix.with_suffix('.events.jsonl')).splitlines()]
            require(c['result']['request_sha256']==hashlib.sha256(raw(prefix.with_suffix('.request.json'))).hexdigest(), 'native request hash mismatch')
            clients.append(c)
        require([c['round'] for c in clients]==[label for label in ['cold','warm1','warm2'] for _ in range(count)], 'wrong client round assignment')
        events=[json.loads(json.loads(s)['line']) for s in raw(folder/'server/worker-stdout.jsonl').splitlines()]
        require(read(folder/'server/worker-exit.json')==dict(returncode=0,expected_close=True), 'actual native worker failure')
        report=check(events,clients,controls,profile,count)
        results.append(dict(label=wave['label'],case=case,**report))
        datasets.append((events,clients,profile,count))
    rejected=[]
    mutations={
        'wrong_token':lambda e,c,p:c[0]['sse_ids'].__setitem__(0,-1),
        'wrong_prompt':lambda e,c,p:c[0]['final_prompt_ids'].__setitem__(0,-1),
        'fake_cache_hit':lambda e,c,p:c[0]['result'].update(cached_prompt_tokens=1),
        'wrong_epoch':lambda e,c,p:next(x for x in e if x['event']=='native_batch')['requests'][0].update(epoch=0),
        'missing_half_dispatch':lambda e,c,p:next(x for x in e if x['event']=='native_chunk_half_attention' and x['stage']=='prefill').update(dispatches=0),
        'mps_fallback':lambda e,c,p:next(x for x in e if x['event']=='native_chunk_attention_mps' and x['stage']=='prefill').update(dispatches=32),
        'owner_leak':lambda e,c,p:e[-1]['memory'].update(garnet_live_matrix_owners=1),
        'false_ttft':lambda e,c,p:c[0]['result'].update(ttft_content_seconds=.001),
        'false_decode':lambda e,c,p:c[0]['result'].update(decode_tokens_per_second=1000.),
        'false_occupancy':lambda e,c,p:next(x for x in e if x['event']=='native_batch').update(active_rows=[0,0,0,0]),
    }
    for label,mutate in mutations.items():
        events,clients,profile,count=copy.deepcopy(datasets[1]);mutate(events,clients,profile)
        try:check(events,clients,controls,profile,count)
        except (ValueError,KeyError,IndexError,StopIteration,TypeError):rejected.append(label)
        else:raise ValueError('Accepted corrupt evidence '+label)
    native={case:{group:summarize([x['metrics'][label] for x in results if x['case']==case for label in labels]) for group,labels in [('cold',['cold']),('warmed',['warm1','warm2'])]} for case in plans}
    references={}
    frozen=read(M/'mistral-repeated-reference-provenance1.json')['files']
    for engine in ['vllm-metal','sglang-mlx']:
        audit=read(M/'reference-campaign1'/(engine+'.audit.json'));require(not audit['issues'] and audit['status']['completed_cohorts']==3, 'unqualified frozen reference collection')
        references[engine]={}
        for case in plans:
            waves=sorted((M/'reference-campaign1'/engine).glob('cohort-*/'+case+'-measured-*'))
            require(len(waves)==15, 'missing frozen measured waves')
            values=[reference_wave(p,bodies[case],prompts[case]) for p in waves]
            references[engine][case]=summarize(values)
    for path,digest in list(files.items()):
        if '/reference-campaign1/' in path:
            require(frozen.get(path)==digest, 'changed/unbound frozen reference artifact '+path)
    ratios={}
    for engine in references:
        ratios[engine]={}
        for case in plans:
            a=native[case]['warmed']['median'];b=references[engine][case]['median']
            ratios[engine][case]=dict(ttft_native_over_reference=a['ttft_median_seconds']/b['ttft_median_seconds'],decode_native_over_reference=a['decode_median_tps']/b['decode_median_tps'],aggregate_native_over_reference=a['aggregate_tps']/b['aggregate_tps'])
    comparison=dict(scope='Mistral7B 4bit,~2048 frozen prompt tokens (C1:2048; C4:2048/2051/2048/2049),128 output tokens; C1 and C4, no prefix retention. Native2 fresh engines/cell:2 first rounds and4 warmed rounds. Frozen references15 warmed measured waves/cell, never rerun. Thermal/OS conditions uncontrolled; native FP32 canonical KV/decode with half prefill, references FP16 state. Cold is fresh process, not hardware cold. Observational, no causal speedup or coding-quality qualification.',native=native,references=references,warmed_ratios=ratios,quality_gate_passed=False,strict_coding='0/8',goal_complete=False)
    report=dict(passed=True,requests=30,output_tokens=3840,all_native_ids_and_text_exact_frozen_fp32=True,negative_controls_rejected=rejected,waves=results,comparison=comparison,quality_gate_passed=False,goal_complete=False,files_sha256=files)
    destination=Path(sys.argv[2]) if len(sys.argv)>2 else ROOT/'serving-audit1.json'
    require(not destination.exists(), 'Never overwrite an audit')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))

if __name__=='__main__':main()
