"""Full-vocabulary MTP evidence plus unchanged full4K target exactness; CPU only."""
from .b4_full4k_exact_gate import admit as target_admit
def admit(selection,reference,candidate,read_reference,read_candidate):
    assert candidate['mtp_study_class']=='CUDAMTPPrefill16B4StudySession'
    m=candidate['mtp_memory'];assert (m['physical_batch'],m['target_prefill_chunk'],m['draft_prefill_chunk'],m['draft_decode_chunk'])==(4,32,16,4)
    assert m['draft_decode_builder_workspace_bytes']==256<<20 and m['draft_prefill_builder_workspace_bytes']==256<<20 and m['extra_draft_engine_weights_duplicated'] is True
    assert m['mtp_graph']['builder_workspace_limit_bytes']==m['prefill16_graph']['builder_workspace_limit_bytes']==256<<20
    routes=[]
    for row in candidate['rows']:
        metrics=row['speculative_metrics'];iterations=metrics['iterations'];assert isinstance(iterations,list) and iterations
        for name in ('target_verification_calls','proposed_draft_tokens','accepted_draft_tokens','mtp_prefill_calls','target_prefix_recomputes','target_prefix_replays'):assert type(metrics[name]) is int and metrics[name]>=0
        assert metrics['mtp_prefill_calls']==249 and metrics['target_verification_calls']==len(iterations) and metrics['proposed_draft_tokens']>0 and metrics['accepted_draft_tokens']<=metrics['proposed_draft_tokens']
        assert metrics['target_prefix_recomputes']==0
        counts=[0]*4;sampled=[0]*4
        for step in iterations:
            a,b=step['processed_target_counts'],step['sampled_so_far'];assert len(a)==len(b)==4 and all(type(n) is int and 0<=n<=4 for n in a) and all(type(n) is int and 0<=n<=128 for n in b) and all(x<=y for x,y in zip(sampled,b))
            counts=[x+y for x,y in zip(counts,a)];sampled=list(b)
        assert counts==[127]*4,'All128 outputs require127 target inputs committed per row'
        routes.append(dict(label=row['label'],verification_calls=metrics['target_verification_calls'],proposed_draft_tokens=metrics['proposed_draft_tokens'],accepted_draft_tokens=metrics['accepted_draft_tokens'],target_prefix_replays=metrics['target_prefix_replays'],committed_target_tokens=counts))
    target=target_admit(selection,reference,candidate,read_reference,read_candidate)
    return dict(status='B4_MTP_prefill16_full4k_target_state_vectors_and128_IDs_exact',target=target,actual_MTP_routes=routes,short_prefix_permutation_tail_repeat_quality_HTTP_or_performance_qualified=False,scope='Fullcontext3groups only; requires separatelyqualifiedsource/DLL/actualgraph/route andindependentB1oracles. Targetexactness+MTPcounterproof alone doesnotqualifyallpartialprefix cases, broadquality, liveHTTP orspeed.')
