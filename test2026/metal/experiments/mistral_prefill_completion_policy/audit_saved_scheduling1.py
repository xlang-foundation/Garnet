"""Inspect frozen first-token distribution and native plans, without inference."""
from pathlib import Path
import collections,copy,hashlib,json,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent;M=ROOT.parent
read=lambda p:json.loads(p.read_text())
files={}
def bind(p):
    files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return read(p)
def need(v,m):
    if not v:raise ValueError(m)
def verify_wave(items):
    need(len(items)==4 and all(e['status']=='completed' and e['generated_tokens']==128 and e['cached_prompt_tokens']==0 for e in items),'wrong frozen wave')
    need(sorted(e['prompt_tokens'] for e in items)==[2048,2048,2049,2051],'wrong frozen prompt lengths')
    for e in items:
        need(abs((e['first_content']-e['submitted'])-e['ttft_content_seconds'])<1e-8,'false TTFT')
    submitted=[e['submitted'] for e in items];ranked=sorted(e['ttft_content_seconds'] for e in items)
    return dict(ranked_ttft_seconds=ranked,first_token_spread_seconds=ranked[-1]-ranked[0],submission_spread_seconds=max(submitted)-min(submitted),wave_median_ttft_seconds=statistics.median(ranked))
engines={};source_engines={}
for engine,source in [('vllm-metal','engine-0.json'),('sglang-mlx','engine-1.json')]:
    config=bind(M/'reference-campaign1/sources'/source);source_engines[engine]=config
    need(config['prefill_budget']==512 and config['max_active_requests']==4 and config['prefix_caching'] is False,'different reference bounds')
    dirs=sorted((M/'reference-campaign1'/engine).glob('cohort-*/p2048-c4-o128-measured-*'));need(len(dirs)==15,'missing reference wave')
    engines[engine]=[verify_wave([bind(p) for p in sorted(d.glob('request-*.result.json'))]) for d in dirs]
native=[];plans=[]
for wave in [2,3]:
    base=M/'mistral-fused-mlp-serving1'/('wave-'+str(wave)+'-p2048-c4-o128')
    profile=bind(base/'profile.json');need(profile['prefill_chunk']==128 and profile['global_prefill_token_budget']==512 and not profile['prefix_cache'],'different native bounds')
    for warm in ['warm1','warm2']:
        native.append(verify_wave([bind(p) for p in sorted((base/'campaign/observed'/warm).glob('*/request-0.result.json'))]))
    p=base/'campaign/observed/server/worker-stdout.jsonl';files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    events=[json.loads(json.loads(line)['line']) for line in p.read_text().splitlines()]
    prefills=[e for e in events if e.get('event')=='native_batch' and e.get('stage')=='prefill']
    need(prefills and all(e['prefill_chunk']==128 and len(e['requests'])<=4 and sum(r['prefill_tokens'] for r in e['requests'])<=512 for e in prefills),'wrong recorded native chunks')
    complete=[e for e in prefills if len(e['requests'])==4 and all(r['prefill_tokens']==128 for r in e['requests'])]
    need(len(complete)>=45,'missing full four-owner prefill steps')
    plans.append(dict(wave=wave,prefill_steps=len(prefills),full_four_owner_128_steps=len(complete),request_count_histogram=dict(collections.Counter(len(e['requests']) for e in prefills))))
engines['garnet-native']=native
summary={name:dict(measured_waves=len(waves),median_ttft_by_rank_seconds=[statistics.median(w['ranked_ttft_seconds'][i] for w in waves) for i in range(4)],median_first_token_spread_seconds=statistics.median(w['first_token_spread_seconds'] for w in waves),median_submission_spread_seconds=statistics.median(w['submission_spread_seconds'] for w in waves),median_of_wave_medians_seconds=statistics.median(w['wave_median_ttft_seconds'] for w in waves)) for name,waves in engines.items()}
rejected=[];source=bind(M/'reference-campaign1/vllm-metal/cohort-1/p2048-c4-o128-measured-0/request-0.result.json')
original=[bind(p) for p in sorted((M/'reference-campaign1/vllm-metal/cohort-1/p2048-c4-o128-measured-0').glob('request-*.result.json'))]
for name,mut in [('missing_request',lambda x:x.pop()),('false_ttft',lambda x:x[0].update(ttft_content_seconds=0)),('cache_hit',lambda x:x[0].update(cached_prompt_tokens=128)),('wrong_prompt',lambda x:x[0].update(prompt_tokens=100))]:
    d=copy.deepcopy(original);mut(d)
    try:verify_wave(d)
    except ValueError:rejected.append(name)
    else:raise ValueError('admitted '+name)
result=dict(passed=True,scope=__doc__,files_sha256=files,summary=summary,native_plan_summary=plans,negative_controls_rejected=rejected,
    inference='The saved staggered reference first-token distribution and clustered native distribution motivate a completion-first native scheduling experiment. These observations alone do not prove the reference internal schedule or quantify causal kernel/prefill speed.',
    all_engines_global_prefill_budget=512,reference_internal_schedule_observed=False,reference_per_operator_kernel_observed=False,
    native_policy_candidate='Optional bounded single prefill owner, resume it before other queued prefills, retain decode batches first, unchanged prepared128-token kernels. Report median/tail TTFT, inter-token latency and aggregate throughput together.',
    no_inference_executed=True,serving_speed_claim=False,goal_complete=False)
p=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'saved-scheduling-audit1.json';need(not p.exists(),'never overwrite');p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,summary=summary,native_plan_summary=plans),indent=2))
