"""Explicit correct-T32 admission and separately named failed-Q-only control."""
from .q128_model_layout import check_graphs as q_check_graphs
O='native_fp8_o128_canonical_projections'
def reference(tokens,q_enabled,o_enabled):
    assert type(tokens) is int and tokens in (32,128) and type(q_enabled) is type(o_enabled) is bool
    if tokens==32:
        assert not q_enabled and not o_enabled
    else:assert q_enabled
    if tokens==128 and not o_enabled:return 'q128-model-admission-attempt1','enabled128',False
    return 'kv128-prefix-cache-attempt1','baseline',True
def check_graphs(graphs,tokens,enabled,q_enabled,o_enabled):
    reference(tokens,q_enabled,o_enabled);q_check_graphs(graphs,tokens,enabled,q_enabled)
    for chunk,row in graphs.items():
        expected=16 if o_enabled and chunk=='128' else 0
        assert type(row[O]) is int and row[O]==expected,(chunk,O,row.get(O),expected)
