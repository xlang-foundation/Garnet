"""Same GDN layout and counters, with the separately qualified reconstructed route."""
from .gdn_chunk_stage_layout import STAGES,layout,pieces,check_graph as original_check
def check_graph(stats,tokens,diagnostic):
    original_check(stats,tokens,diagnostic)
    expected={'native_fp8_z128_canonical_projections':0,'native_fp8_z128_reconstructed_projections':int(tokens==128),'native_fp8_kv128_canonical_projections':0}
    for key,value in expected.items():assert type(stats[key]) is int and stats[key]==value,(key,stats.get(key))
    assert stats['native_fp8_z128_canonical_projections']+stats['native_fp8_z128_reconstructed_projections']==int(tokens==128)
    return True
