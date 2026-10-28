"""Repeated Garnet profiles versus historical engines on one equal-work subset."""
import argparse,hashlib,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path)
p.add_argument('--model',choices=('2b','4b'),required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--vllm-extra-suffix',action='append',default=[])
feature=p.add_mutually_exclusive_group();feature.add_argument('--joined-decode',action='store_true');feature.add_argument('--activation-storage',action='store_true');feature.add_argument('--internal-stores',action='store_true');feature.add_argument('--vision-graph',action='store_true');feature.add_argument('--compact-kv',action='store_true')
p.add_argument('--study-name');p.add_argument('--observer-prefix');a=p.parse_args()
assert not a.output.exists(),'Preserve existing summary'
root=a.root;study_name=a.study_name or ('compact-kv' if a.compact_kv else 'vision-graph' if a.vision_graph else 'internal-stores' if a.internal_stores else 'activation-arena' if a.activation_storage else 'joined-decode' if a.joined_decode else 'promptpack-reusehead')
study_path=root/f'packed-study-{a.model}-b{a.batch}-{study_name}-summary.json'
study=json.loads(study_path.read_text());assert study['all28_workload_owner_controls_passed'] if a.compact_kv else study['same_all_28_input_prompt_output_stop_signatures']
role_sources=({role:{'file':f'matched512-garnet-{a.model}-b{a.batch}-study-{study_name}-{role}.json',
    'sha256':study['sources_sha256'][str(root/f'matched512-garnet-{a.model}-b{a.batch}-study-{study_name}-{role}.json')]}
    for role in ('a1','b1','b2','a2')} if a.compact_kv else study['sources'])
rows={};sources={}
for role,record in role_sources.items():
    path=root/record['file'];blob=path.read_bytes();digest=hashlib.sha256(blob).hexdigest()
    assert digest==record['sha256'],'Raw study changed after preservation summary'
    rows[role]={(r['trial'],r['workload_group']):r for r in json.loads(blob)}
    assert len(rows[role])==28;sources[role]={'file':path.name,'sha256':digest}
external={'lmstudio':f'matched512-lmstudio-{a.model}-b{a.batch}-partitioned.json',
    'vllm_serial':f'matched512-vllm-{a.model}-b{a.batch}.json'}
for suffix in a.vllm_extra_suffix:external['vllm'+suffix]=f'matched512-vllm-{a.model}-b{a.batch}{suffix}.json'
for name,filename in external.items():
    blob=(root/filename).read_bytes();rows[name]={(r['trial'],r['workload_group']):r for r in json.loads(blob)}
    assert len(rows[name])==28;sources[name]={'file':filename,'sha256':hashlib.sha256(blob).hexdigest()}
base=rows['a1']
for name,data in rows.items():
    assert data.keys()==base.keys()
    for key,row in data.items():
        assert row['input_sha256']==base[key]['input_sha256'] and row['batch']==a.batch
        assert row['warmup']==base[key]['warmup']
        if name.startswith('vllm'):assert [q['prompt_token_ids'] for q in row['records']]==base[key]['prompt_token_ids']
def counts(name,row):
    if name in role_sources:return row['sampled_tokens_including_stop']
    if name=='lmstudio':return [q['result']['stats']['total_output_tokens'] for q in row['records']]
    return [len(q['token_ids']) for q in row['records']]
keys=[key for key,row in base.items() if not row['warmup'] and all(counts(name,data[key])==[32]*a.batch for name,data in rows.items())]
assert keys,'No common equal32 groups'
med=statistics.median
letters='ab' if a.joined_decode or a.activation_storage or a.internal_stores or a.vision_graph or a.compact_kv else 'abc';selected='garnet_b' if a.joined_decode or a.activation_storage or a.internal_stores or a.vision_graph or a.compact_kv else 'garnet_c'
latencies={'garnet_'+letter:med(med(rows[letter+repeat][key]['wall_seconds'] for repeat in ('1','2')) for key in keys)*1000 for letter in letters}
latencies.update({name:med(rows[name][key]['wall_seconds'] for key in keys)*1000 for name in external})
fastest=min(external,key=lambda name:latencies[name]);threshold=latencies[fastest]/2
observation_file=None
if a.compact_kv:
    assert a.observer_prefix,'Compact timing needs its actual observed-attempt prefix'
    observation_path=root/f'{a.observer_prefix}-{a.model}-b{a.batch}-process-audit.json'
    observation_blob=observation_path.read_bytes();observation=json.loads(observation_blob)
    assert observation['all_four_roles_covered'] and observation['known_overlap_count']==0 and not observation['observer_metadata_errors']
    for filename,expected in observation['sources_sha256'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected
    observation_file=observation_path.name
    sources['process_observation_audit']={'file':observation_file,'sha256':hashlib.sha256(observation_blob).hexdigest()}
result={'model':a.model,'batch':a.batch,'sources':sources,'study_summary':study_path.name,'common_equal32_groups':len(keys),
    'garnet_dll_sha256':study['garnet_dll_sha256'],'process_observation_audit_file':observation_file,
    'group_keys':keys,'median_group_ms':latencies,'fastest_tested_baseline':fastest,'half_fastest_baseline_ms':threshold,
    selected+'_speed_ratio_vs_fastest':latencies[fastest]/latencies[selected],
    'latency_threshold_met':latencies[selected]<=threshold,'independent_quality_accepted':False,'goal_2x_achieved':False,
    'garnet_common_group_derived_post_first_sample_tokens_per_second':{
        'garnet_'+letter:med(med((sum(rows[letter+repeat][key]['sampled_tokens_including_stop'])-a.batch)/rows[letter+repeat][key]['decode_seconds'] for repeat in ('1','2')) for key in keys) for letter in letters},
    'scope':f'One common equal32 subset across all {len(role_sources)} Garnet runs and every listed historical competitor. Garnet first medians two repeats per group, then groups; competitors are single saved cohorts and were not rerun. Garnet measures Model.infer_batch API-return wall; derived decode rate includes host/stop overhead. APIs, intermediate arithmetic and speculative settings differ. LM2B drafting observed, LM4B unresolved. No significance, causal, independent-quality, or exhaustive-baseline claim.'}
a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('sources','group_keys')},indent=2))
