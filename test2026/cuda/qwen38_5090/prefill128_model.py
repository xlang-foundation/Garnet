"""Exact full-model T128 qualification, extending the retained full4K oracle."""
from .tiled_scores_model import replay_tiled_model,same_state

def replay_prefill128_model(report,read,reference,read_reference,cases):
    assert type(report['target128_enabled']) is bool and report['target128_enabled'] is True
    for key,value in (('target_prefill_chunk',128),('target_decode_chunk',4),('draft_prefill_chunk',16)):
        assert type(report[key]) is int and report[key]==value
    assert report['target_program_chunks']==[4,128] and report['tiled_scores_enabled'] is False
    prefill=report['prefill_helper'];decode=report['decode_helper'];memory=report['memory_after_construction']
    for helper,chunk in ((prefill,128),(decode,4)):
        assert helper['ready'] is True and helper['poisoned'] is False
        assert type(helper['batch']) is int and helper['batch']==1 and type(helper['chunk']) is int and helper['chunk']==chunk
        assert helper['canonical_state_width']==106823680 and helper['linear_replay_width']==39714816
    assert prefill['target_width']==44564480
    assert memory['target_prefill_chunk']==128 and memory['draft_prefill_chunk']==16 and memory['draft_decode_chunk']==4
    assert memory['study_gpu_qualified'] is memory['study_performance_admitted'] is False
    checked=replay_tiled_model(report,read,reference,read_reference,cases)
    assert len(report['ordinary_observations'])==len(report['cases'])==6
    for actual,full in zip(report['ordinary_observations'],report['cases']):
        assert set(actual)=={'sequence','case_index','key','request_sha256','image_sha256','prompt_ids','ordinary_ids','reset_state','ordinary_prefill','ordinary_final'}
        for key in ('sequence','case_index','key','request_sha256','image_sha256','prompt_ids','ordinary_ids'):assert actual[key]==full[key]
        for key in ('reset_state','ordinary_prefill','ordinary_final'):same_state(actual[key],full[key],read,read)
    return {'status':'prefill128_full4k_retained_target_state_exact','cases':checked['cases'],'captured_states':32,'complete_retained_baseline_states_and_raw_vectors_exact':True,'target_prefill_chunk':128,'target_decode_chunk':4,'draft_prefill_chunk':16,'ordinary_mtp_exact':True,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False}
