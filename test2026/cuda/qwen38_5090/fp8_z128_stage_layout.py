"""Pure embedded-compatible Z stage/counter and separate-process policy checks."""
from .gdn_chunk_stage_layout import STAGES,layout,pieces,check_graph as previous_check_graph
POLICY_VALUES=(None,'','0','1','true','01',' 1','1 ','2','-1')


def check_graph(stats,tokens,diagnostic):
    previous_check_graph(stats,tokens,diagnostic)
    count=stats['native_fp8_z128_canonical_projections']
    assert type(count) is int and count==int(tokens==128)
    return True


def check_policy_report(report,expected):
    assert expected in POLICY_VALUES and report['status']=='fp8_z128_policy_control_observed'
    assert report['graph_inference_executed'] is report['model_or_performance_qualified'] is False
    actual=report['policy_value']
    if expected in (None,''):assert actual in (None,'')
    else:assert actual==expected
    invalid=expected not in (None,'','0','1');assert type(report['rejected']) is bool and report['rejected'] is invalid
    if invalid:
        assert type(report['error']) is str and 'GARNET_FP8_GDN_Z128_CANONICAL' in report['error'] and 'must be0 or1' in report['error'] and report['graph'] is None
    else:
        assert report['error'] is None;stats=report['graph']
        assert stats['checkpoint_weight_precision']=='modelopt_mixed'
        values=(('native_nvfp4_projection_weights',2),('native_fp8_projection_weights',3),('external_compressed_weight_inputs',5),('native_int4_g128_projection_weights',0),('native_fp8_m128_canonical_projections',1),('native_fp8_z128_canonical_projections',int(expected=='1')))
        for key,value in values:assert type(stats[key]) is int and stats[key]==value
    return {'expected_environment_value':expected,'observed_environment_value':actual,'rejected':invalid,'native_Z128_selected':expected=='1','graph_inference_executed':False,'native_frontend_exercised':True}
