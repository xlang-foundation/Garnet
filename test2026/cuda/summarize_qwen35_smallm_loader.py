"""Recompute exact eligibility, shared selection and heldout component gates."""
import datetime
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1]).resolve();output=root/'qwen35-smallm-loader-v1-summary.json';assert not output.exists()
sources={}

def read(path):
    raw=path.read_bytes();sources[path.name]=hashlib.sha256(raw).hexdigest();return json.loads(raw.decode('utf-8-sig'))

def validate_math(records,exact):
    for r in records:
        assert r['finite_words']>0 and r['fp64_samples_per_provider']==128
        assert 0<=r['maximum_original_fixed_bound_fraction']<=1
        if exact:assert r['different_bf16_words']==0

def pairs(record,count):
    for key in ('graph_us','excluded_eviction_us','including_eviction_total_us'):
        assert len(record[key])==2
        assert all(len(v)==count and all(math.isfinite(x) and x>=0 for x in v) for v in record[key])
    a,b=(statistics.median(v) for v in record['graph_us']);assert a>0 and b>0
    assert abs(a-record['baseline_median_us'])<1e-7 and abs(b-record['candidate_median_us'])<1e-7
    assert abs(a/b-record['ordinary_over_candidate'])<1e-7
    return a,b

manifest=read(root/'qwen35-smallm-loader-v1-fixtures.json');design=read(root/'qwen35-smallm-loader-v1-prospective-design.json')
identity_path=root/'qwen35-smallm-loader-v1-before-build-identity.json';identity=read(identity_path);epoch=sources[identity_path.name]
build=read(root/'qwen35-smallm-loader-v1-build-exit.json');assert build['exit_code']==0
binary=read(root/'qwen35-smallm-loader-v1-executable-identity.json')
assert binary['execution_identity_sha256'].lower()==epoch
observer=root/'qwen35-smallm-loader-v1-processes.jsonl';blob=observer.read_bytes();sources[observer.name]=hashlib.sha256(blob).hexdigest()
obs=[json.loads(line) for line in blob.decode('utf-8-sig').splitlines()]
assert obs[0]['event']=='observer_started' and obs[-1]['event']=='stopped'
time=lambda s:datetime.datetime.fromisoformat(s).timestamp()
processes=[r for r in obs if r['event']=='process_observed'];errors=[r for r in obs if r['event'] in ('observer_failure','process_metadata_unavailable')]
results=[];previous_end=None
for group in manifest['groups']:
    name=group['name'];stem='qwen35-smallm-loader-v1-'+name
    terminal=read(root/(stem+'-exit.json'));assert terminal['exit_code']==0
    assert terminal['execution_identity_sha256'].lower()==epoch
    assert terminal['binary_sha256'].lower()==binary['sha256'].lower()
    assert terminal['gpu_uuid']==design['gpu_uuid']
    path=root/(stem+'.jsonl');raw=path.read_bytes();sources[path.name]=hashlib.sha256(raw).hexdigest()
    rows=[json.loads(line) for line in raw.decode().splitlines()];meta=rows[0];done=rows[-1]
    assert meta['event']=='metadata' and meta['group']==group and meta['gpu_uuid']==design['gpu_uuid']
    assert meta['selection_seed']==384771 and meta['holdout_seed']==384772
    assert meta['workspace_bytes']==0 and meta['selection_pairs']==5 and meta['holdout_pairs']==31
    assert meta['observer_startup_hold_ms']==1250
    assert meta['eviction_bytes']>=max(256<<20,4*meta['l2_bytes'])
    assert [f['name'] for f in meta['fixtures']]==group['fixtures']
    assert meta['fixtures']==[next(f for f in manifest['fixtures'] if f['name']==name) for name in group['fixtures']]
    assert len(meta['baseline_math'])==len(group['fixtures'])
    all_math=list(meta['baseline_math'])
    for record in rows:
        if record['event'] in ('selection_only','selection_bit_rejected'):
            assert len(record['math'])==len(group['fixtures']);all_math+=record['math']
        if record['event']=='selection_only':
            assert len(record['replay_math'])==len(group['fixtures']);all_math+=record['replay_math']
        if record['event']=='holdout_correctness':
            assert len(record['matrices'])==len(group['fixtures'])
            for matrix in record['matrices']:all_math += [matrix['baseline'],matrix['candidate']]
        if record['event']=='holdout':all_math.append(record['replay_math'])
    assert all(r['finite_words']==group['rows']*group['columns'] for r in all_math)
    validate_math(meta['baseline_math'],True)
    assert done['event']=='complete' and done['all_guards_inputs_weights_preserved']
    assert not done['production_model_modified'] and not done['full_request_accepted']
    selection=[r for r in rows if r['event']=='selection_only'];rejected=[r for r in rows if r['event']=='selection_bit_rejected']
    assert done['configurations_checked']==(9 if group['tokens']==16 else 6)==len(selection)+len(rejected)
    assert done['selection_bit_rejections']==len(rejected) and done['exact_candidates']==len(selection)
    configs=[]
    for record in rejected:
        validate_math(record['math'],False);assert any(r['different_bf16_words']>0 for r in record['math']);configs.append(record['config'])
    for record in selection:
        configs.append(record['config']);validate_math(record['math'],True);validate_math(record['replay_math'],True)
        assert len(record['cold_samples'])==len(group['fixtures'])
        baseline_cost=candidate_cost=0
        for count,sample,fixture in zip(group['counts_per_pass'],record['cold_samples'],group['fixtures']):
            assert sample['fixture']==fixture;a,b=pairs(sample['samples'],5);baseline_cost+=count*a;candidate_cost+=count*b
        assert abs(record['weighted_cold_ratio']-baseline_cost/candidate_cost)<1e-7
    assert len({json.dumps(c,sort_keys=True) for c in configs})==len(configs)
    for config in configs:
        assert config['tokens']==group['tokens'] and config['tile_k'] in (32,64,128) and config['warps'] in (2,4,8)
        assert config['warps']>config['tokens']//16
    holdouts=[]
    if not selection:
        assert done['screen_status']=='rejected_no_exact_candidate' and not done['holdout_performed']
        assert not any(r['event'] in ('selection_frozen','holdout_correctness','holdout') for r in rows)
    else:
        frozen=[r for r in rows if r['event']=='selection_frozen'];assert len(frozen)==1;frozen=frozen[0]
        winner=sorted(selection,key=lambda r:-r['weighted_cold_ratio'])[0]
        assert frozen['config']==winner['config'] and frozen['selection_cold_ratio']==winner['weighted_cold_ratio']
        assert frozen['gate_up_share_one_geometry'] and not frozen['other_finalists_allowed_for_rescue']
        assert frozen['selection_input_and_weight_bytes_preserved']
        fresh=[r for r in rows if r['event']=='holdout_correctness'];assert len(fresh)==1;fresh=fresh[0]
        assert rows.index(frozen)<rows.index(fresh)
        assert [r['fixture'] for r in fresh['matrices']]==group['fixtures']
        exact=all(r['candidate']['different_bf16_words']==0 for r in fresh['matrices'])
        assert fresh['all_output_words_bit_exact']==exact
        for r in fresh['matrices']:validate_math([r['baseline']],True);validate_math([r['candidate']],exact)
        raw_holdouts=[r for r in rows if r['event']=='holdout']
        if not exact:
            assert done['screen_status']=='rejected_fresh_input_words' and not done['holdout_performed'] and not raw_holdouts
        else:
            assert done['screen_status']=='holdout_completed' and done['holdout_performed']
            assert [r['fixture'] for r in raw_holdouts]==group['fixtures']
            for record,count in zip(raw_holdouts,group['counts_per_pass']):
                assert record['config']==frozen['config'] and record['all_output_words_bit_exact'];validate_math([record['replay_math']],True)
                a,b=pairs(record['cold'],31);wa,wb=pairs(record['warm'],31)
                holdouts.append({'fixture':record['fixture'],'count_per_pass':count,'baseline_cold_us':a,'candidate_cold_us':b,
                    'cold_ratio':a/b,'warm_ratio':wa/wb,'config':frozen['config']})
    lo,hi=map(time,(terminal['started_utc'],terminal['ended_utc']));assert previous_end is None or previous_end<=lo;previous_end=hi
    assert time(obs[0]['utc'])<=lo<hi<=time(obs[-1]['utc'])
    own=next(r for r in processes if r['pid']==meta['process_id'] and lo<=time(r['observed_utc'])<=hi)
    foreign=[]
    for snap in obs:
        if snap['event']!='snapshot' or max(lo,time(snap['begin_utc']))>min(hi,time(snap['end_utc'])):continue
        other=[p for p in snap['processes'] if p['pid']!=own['pid']]
        if other:foreign.append({'begin_utc':snap['begin_utc'],'end_utc':snap['end_utc'],'processes':other})
    phase_errors=[r for r in errors if 'observed_utc' not in r or lo<=time(r['observed_utc'])<=hi]
    results.append({'group':name,'model':group['model'],'status':done['screen_status'],'exact_candidates':len(selection),
        'bit_rejected_candidates':len(rejected),'holdouts':holdouts,'process_id':own['pid'],
        'observed_foreign_process_snapshots':foreign,'observer_metadata_errors':phase_errors,
        'process_observation_qualified':not foreign and not phase_errors})
models=[]
for size in ('2b','4b'):
    groups=[r for r in results if r['model']==size];assert len(groups)==3
    complete=all(r['status']=='holdout_completed' for r in groups)
    values=[v for r in groups for v in r['holdouts']]
    ratio=(sum(v['count_per_pass']*v['baseline_cold_us'] for v in values)/sum(v['count_per_pass']*v['candidate_cold_us'] for v in values)) if complete else None
    no_regression=complete and all(v['cold_ratio']>=1/1.05 for v in values)
    models.append({'model':size,'all_four_matrices_fresh_input_bit_exact':complete,'operator_weighted_cold_ratio':ratio,
        'no_operator_latency_regression_above5percent':no_regression,'component_speed_gate_passed':complete and ratio>=1.10 and no_regression})
qualified=all(r['process_observation_qualified'] for r in results)
result={'all_six_completed_case_protocols_audited':True,'groups':results,'models':models,
    'permits_only_further_component_validation':qualified and all(m['component_speed_gate_passed'] for m in models),
    'process_observation_qualified':qualified,'execution_identity_sha256':epoch,'binary_sha256':binary['sha256'],
    'maximum_snapshot_gap_seconds':obs[-1]['maximum_snapshot_gap_seconds'],'sources_sha256':sources,
    'production_promoted':False,'model_memory_accepted':False,'model_quality_accepted':False,'full_request_accepted':False,
    'scope':'Six sequential learned-shape groups/eight matrices, bitexact eligibility, five-pair shape-shared selection, one frozen geometry before fresh-input31cold/31warm holdout. No finalist rescue. Component gate only; no whole-model latency/memory/quality, cache-residency, all-batch/capacity, sanitizer or2x acceptance.'}
output.write_text(json.dumps(result,indent=2)+'\n')
print('QWEN35_SMALLM_AUDITED_COMPONENT_GATE',result['permits_only_further_component_validation'])
