"""Require both native policies, then the unchanged strict retained full4K oracle."""
from .prefill128_fp8_model import validate_targets as original_targets,validate_selection as original_selection,replay_prefill128_fp8_model
FIELD='native_fp8_z128_canonical_projections'

def count(graph,expected):
    assert isinstance(graph,dict) and type(graph.get(FIELD)) is int and graph[FIELD]==expected
    return graph[FIELD]

def validate_targets(graphs,policy):
    old=original_targets(graphs,policy)
    return {'m128':old,'z128':{'4':count(graphs['4'],0),'128':count(graphs['128'],48 if policy=='1' else 0)}}

def validate_selection(report):
    old=original_selection(report);assert report['fp8_z128_policy']==report['fp8_m128_policy']=='1' and type(report['fp8_z128_policy']) is str
    targets=validate_targets(report['fp8_target_graphs'],report['fp8_z128_policy']);aux=report['fp8_auxiliary_graphs']
    values={name:count(aux[name],0) for name in ('draft4','draft16','vision')}
    values['prefill_norm']=None if aux['prefill_norm'] is None else count(aux['prefill_norm'],0)
    return {**old,'z128_target_counts':targets['z128'],'z128_auxiliary_counts':values}

def replay_full4k_model(report,read,reference,read_reference,cases):
    selection=validate_selection(report)
    checked=replay_prefill128_fp8_model(report,read,reference,read_reference,cases)
    return {**checked,'fp8_z128_selection':selection}
