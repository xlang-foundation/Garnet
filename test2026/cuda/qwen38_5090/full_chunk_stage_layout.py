"""Embedded stdlib-only metadata for actual layer3 ordinary chunk observations."""
from .kv128_prefix_cache_witness import check_graphs as original_check_graphs,parts
STAGES=(('input',5120),('normalized',5120),('qg',12288),('key',1024),('value',1024),('core',6144),('attention',5120),('residual',5120),('mlp_input',5120),('output',5120))
assert sum(w for _,w in STAGES)==51200
def check_graphs(graphs,tokens,enabled):
    assert type(enabled) is bool;original_check_graphs(graphs,tokens)
    for chunk,row in graphs.items():
        expected=1 if enabled and chunk==str(tokens) else 0
        assert type(row['native_full_chunk_stage_witnesses']) is int and row['native_full_chunk_stage_witnesses']==expected
def stage_offsets(tokens,prefix):
    assert type(tokens) is int and tokens in (32,128) and type(prefix) is int and prefix>0
    result=[]
    for name,width in STAGES:result.append((name,prefix,tokens*width));prefix+=tokens*width
    return result
