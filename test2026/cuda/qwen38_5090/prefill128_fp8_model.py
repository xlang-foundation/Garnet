"""Actual native policy selection proof followed by unchanged strict full4K oracle."""
from .prefill128_model import replay_prefill128_model
FIELD='native_fp8_m128_canonical_projections'

def count(graph,expected):
    assert isinstance(graph,dict) and type(graph.get(FIELD)) is int and graph[FIELD]==expected
    return graph[FIELD]

def validate_targets(graphs,policy):
    assert type(policy) is str and policy in ('0','1') and set(graphs)=={'4','128'}
    return {'4':count(graphs['4'],0),'128':count(graphs['128'],48 if policy=='1' else 0)}

def validate_selection(report):
    assert report['fp8_m128_policy']=='1' and type(report['fp8_m128_policy']) is str
    target=validate_targets(report['fp8_target_graphs'],report['fp8_m128_policy']);aux=report['fp8_auxiliary_graphs']
    assert set(aux)=={'draft4','draft16','prefill_norm','vision'}
    counts={name:count(aux[name],0) for name in ('draft4','draft16','vision')}
    counts['prefill_norm']=None if aux['prefill_norm'] is None else count(aux['prefill_norm'],0)
    return {'target_counts':target,'auxiliary_counts':counts,'native_selection_verified':True}

def replay_prefill128_fp8_model(report,read,reference,read_reference,cases):
    selection=validate_selection(report)
    checked=replay_prefill128_model(report,read,reference,read_reference,cases)
    return {**checked,'fp8_m128_selection':selection}
