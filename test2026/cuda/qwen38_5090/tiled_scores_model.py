"""Exact new-DLL/default/tiled versus retained baseline whole-model witnesses."""
from .mtp_prefill16_full4k import replay_report

STATE_NAMES=('reset_state','ordinary_prefill','candidate_prefill','ordinary_final','candidate_final')
def same_state(actual,expected,read,read_reference):
    assert actual['logical']==expected['logical']
    assert set(actual['vectors'])==set(expected['vectors'])=={'hidden','normalized'}
    for kind in ('hidden','normalized'):
        a=actual['vectors'][kind];b=expected['vectors'][kind]
        assert a['bytes']==b['bytes']==20480 and a['sha256']==b['sha256']
        assert read(a['file'])==read_reference(b['file'])

def same_record(actual,expected,read,read_reference):
    for key in ('sequence','case_index','key','request_sha256','image_sha256','prompt_ids','ordinary_ids','candidate_ids','prefill_first_tokens','callbacks','sampled_tokens_including_stop','stop_token_ids'):
        assert actual[key]==expected[key],key
    for name in STATE_NAMES:same_state(actual[name],expected[name],read,read_reference)

def replay_tiled_model(report,read,reference,read_reference,cases):
    actual=replay_report(report,read,cases);baseline=replay_report(reference,read_reference,cases)
    assert actual['captured_states']==baseline['captured_states']==32
    assert type(report['tiled_scores_enabled']) is bool
    same_state(report['empty_state'],reference['empty_state'],read,read_reference)
    same_state(report['final_reset'],reference['final_reset'],read,read_reference)
    assert len(report['cases'])==len(reference['cases'])==6
    for a,b in zip(report['cases'],reference['cases']):same_record(a,b,read,read_reference)
    return {'status':'tiled_scores_full4k_retained_baseline_exact','tiled_scores_enabled':report['tiled_scores_enabled'],'cases':actual['cases'],'captured_states':32,
            'complete_retained_baseline_states_and_raw_vectors_exact':True,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False}
