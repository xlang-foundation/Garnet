"""Audit saved preliminary operator coverage and recompute command medians.
CPU numerical assertions were executed by the pinned C++ host. Output word
parents and raw GPU clock endpoints were not retained, so this saved-data audit
does not independently replay the FP64 oracle or qualify numeric/performance use.
"""
from pathlib import Path
import collections,copy,hashlib,json,math,statistics,sys
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
DEST=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else ROOT/'audit1.json'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def need(v,m):
    if not v:raise ValueError(m)
def check(d):
    need(d['completed'] and d['all_pipelines_prepared_before_collection'] and d['numerically_eligible_for_further_screening'],'incomplete/rejected pilot')
    need(d['numerical_failures']==d['coefficient_reconstruction_failures']==d['fixed1_batch_word_failures']==0,'host correctness failed')
    need(d['oracle_words']==276480 and d['parent_words_checked']==1033088,'changed host check coverage')
    need(d['close_allocated_bytes']<=d['baseline_allocated_bytes']+4194304,'standalone allocations retained')
    cells={(x['shape'],x['rows'],x['method']):x for x in d['cells']};expected={(s,r,m) for s in ['gate','down'] for r in [1,4] for m in [0,1,4]}
    need(set(cells)==expected and len(cells)==len(d['cells']),'missing/extra correctness cell')
    for key,x in cells.items():
        need(math.isfinite(x['maximum_original_fp64_fraction']) and 0<=x['maximum_original_fp64_fraction']<=1,'host original-bound check failed')
        need(len(x['output_sha256'])==64 and x['old_fp32_nonidentical_words']>0 if key[2] else x['old_fp32_nonidentical_words']==0,'wrong arithmetic-policy difference')
    for shape in ['gate','down']:
        for rows in [1,4]:need(cells[(shape,rows,1)]['output_sha256']==cells[(shape,rows,4)]['output_sha256'],'candidate column tilings differ')
    seen=set();buckets=collections.defaultdict(list)
    for x in d['timing']:
        key=(x['shape'],x['rows'],x['method'],x['round']);need(key not in seen and key[:3] in expected and 0<=x['round']<21,'wrong measured coverage');seen.add(key)
        at=(x['position']+x['round']+5)%3
        if (x['round']+5)&1:at=2-at
        need(x['position'] in [0,1,2] and x['method']==[0,1,4][at],'changed paired rotation/order')
        need(type(x['gpu_ms']) in [int,float] and math.isfinite(x['gpu_ms']) and x['gpu_ms']>0,'invalid command GPU duration');buckets[key[:3]].append(x['gpu_ms'])
    need(len(seen)==252 and all(len(v)==21 for v in buckets.values()),'missing samples')
    summaries=[]
    for shape in ['gate','down']:
        for rows in [1,4]:
            med={m:statistics.median(buckets[(shape,rows,m)]) for m in [0,1,4]}
            summaries.append(dict(shape=shape,rows=rows,median_command_ms={str(m):v for m,v in med.items()},control_over_candidate_ratio={str(m):med[0]/med[m] for m in [1,4]}))
    return summaries,cells
def main():
    need(not DEST.exists(),'never overwrite audit');results={};summaries={};cells={};files={}
    for index in [1,2]:
        need(read(ROOT/('run'+str(index)+'-status.json'))['returncode']==0 and read(ROOT/('postflight'+str(index)+'.json'))['passed'],'pilot unfinished')
        for n,h in read(ROOT/('preflight'+str(index)+'.json'))['files_sha256'].items():need(sha(Path(n))==h,'changed bound input '+n)
        p=ROOT/('pilot-output'+str(index))/'result1.json';results[index]=read(p);summaries[str(index)],cells[index]=check(results[index]);files[str(p)]=sha(p)
    need({k:x['output_sha256'] for k,x in cells[1].items()}=={k:x['output_sha256'] for k,x in cells[2].items()},'vector-load revision changed output bits')
    rejected=[]
    for name,mut in [('missing_cell',lambda d:d['cells'].pop()),('missing_sample',lambda d:d['timing'].pop()),('wrong_order',lambda d:d['timing'][0].update(position=9)),('nonfinite_time',lambda d:d['timing'][0].update(gpu_ms=float('nan'))),('wrong_bound',lambda d:d['cells'][0].update(maximum_original_fp64_fraction=2)),('batch_identity_failure',lambda d:d.update(fixed1_batch_word_failures=1)),('retained_buffers',lambda d:d.update(close_allocated_bytes=2**40))]:
        bad=copy.deepcopy(results[1]);mut(bad)
        try:check(bad)
        except (ValueError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise ValueError('corruption admitted '+name)
    result=dict(passed=True,scope=__doc__,pilot_summaries=summaries,candidate_outputs_unchanged_by_vector_load_revision=True,numeric_assertions_host_reported=True,raw_word_parents_retained=False,independent_saved_fp64_replay=False,raw_gpu_clock_endpoints_retained=False,full_trained_operator_qualification=False,repeated_operator_timing_qualification=False,selected_for_native_integration=False,reason='Large four-row regressions in both pilots; do not integrate the new arithmetic policy.',negative_controls_rejected=rejected,files_sha256=files,no_runtime_integration=True,serving_speed_claim=False,goal_complete=False)
    DEST.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
