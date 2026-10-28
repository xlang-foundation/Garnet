"""Embedded stdlib metadata checks; no NumPy or native imports."""
from .fa3_projection_witness_layout import STAGES,weight_metadata,check_graph as parent_check
def check_graph(stats,tokens):
    parent_check(stats,tokens)
    expected=2 if tokens==128 else 0
    assert type(stats['native_fp8_kv128_canonical_projections']) is int and stats['native_fp8_kv128_canonical_projections']==expected
