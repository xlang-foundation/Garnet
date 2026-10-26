"""Bind two successful full-output replays without requiring identical FP64 summaries."""
import copy,json,math

def compare_replays(local,remote):
    a=copy.deepcopy(local);b=copy.deepcopy(remote)
    expected='prefill128_attention_component_full_fp64_replayed'
    assert a['status']==b['status']==expected
    assert a['full_output_scalars_replayed']==b['full_output_scalars_replayed']==28311552
    assert a['numerical_failures']==b['numerical_failures']==[]
    assert len(a['cases'])==len(b['cases'])==36
    differences=[]
    for x,y in zip(a['cases'],b['cases']):
        detail={'id':x['id']}
        for key,limit in (('max_absolute_error',1e-12),('max_tolerance_fraction',1e-8)):
            left=x.pop(key);right=y.pop(key)
            assert type(left) is float and type(right) is float
            assert math.isfinite(left) and math.isfinite(right) and left>=0 and right>=0
            if key=='max_tolerance_fraction':assert left<=1 and right<=1
            assert abs(left-right)<=limit
            if left!=right:detail[key]={'local':left,'remote':right,'absolute_difference':abs(left-right)}
        if len(detail)>1:differences.append(detail)
        assert x==y
    assert json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)
    return {'same_archive_and_verifier':True,'all_non_summary_fields_exact':True,'full_independent_numerical_checks_passed_on_both_machines':True,
            'summary_differences':differences,'summary_values_identical':local==remote,
            'scope':'Only two per-case FP64 error summaries may differ across machines, bounded as recorded. Both original archived checkers independently enforce every scalar at unchanged ATOL3e-5/RTOL2e-5, complete output counts and inactive exact zeros. No GPU output, archive, acceptance predicate, source or workflow differences permitted. Cause of summary variation is not established.'}
