"""Explicit new-family metadata gates; ordinary layer2 math and array layout are unchanged."""
from .layer2_ordinary_layout import layout,pieces,validate_profile,INPUT_SOURCE
from .gdn_chunk_stage_layout import check_graph as common_check_graph
def check_graph(stats,tokens,diagnostic):
    common_check_graph(stats,tokens,diagnostic)
    for key,expected in [('native_fp8_z128_canonical_projections',0),('native_fp8_z128_reconstructed_projections',int(tokens==128))]:
        assert type(stats[key]) is int and stats[key]==expected,(key,stats[key])
    return True
