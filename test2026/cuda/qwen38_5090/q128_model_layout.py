"""Embedded stdlib-only physical Q128 route and unchanged original stage guards."""
from .full_chunk_stage_layout import check_graphs as original_check_graphs
Q='native_fp8_q128_canonical_projections'
def reference_profile(tokens,q_enabled):
    assert type(tokens) is int and tokens in (32,128) and type(q_enabled) is bool and (not q_enabled or tokens==128)
    return 'baseline' if tokens==32 or q_enabled else 'native'
def check_graphs(graphs,tokens,enabled,q_enabled):
    reference_profile(tokens,q_enabled);assert enabled is True;original_check_graphs(graphs,tokens,enabled)
    for chunk,row in graphs.items():
        expected=16 if q_enabled and chunk=='128' else 0
        assert type(row[Q]) is int and row[Q]==expected,(chunk,Q,row.get(Q),expected)
