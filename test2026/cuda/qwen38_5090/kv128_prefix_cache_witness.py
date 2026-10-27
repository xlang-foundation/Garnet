"""Same complete-cache and packed-state oracles with explicit physical route counts."""
from .fa3_prefix_cache_witness import PREFIXES,parts,cache_components,same_state,read_cache,cache_difference,check_graphs as original_check_graphs
OLD='native_fp8_z128_canonical_projections';RECON='native_fp8_z128_reconstructed_projections';KV='native_fp8_kv128_canonical_projections'
def check_graphs(graphs,tokens):
    assert type(tokens) is int and tokens in (32,128) and set(graphs)=={'4',str(tokens)}
    logical={}
    for chunk,row in graphs.items():
        active=tokens==128 and chunk=='128'
        for field,expected in ((OLD,0),(RECON,48 if active else 0),(KV,32 if active else 0)):
            assert type(row[field]) is int and row[field]==expected,(chunk,field,row.get(field))
        logical[chunk]=dict(row)
        # Original logical-Z selection check remains48. Raw physical counts above
        # identify the reconstructed family separately; raw stats are never changed.
        logical[chunk][OLD]=row[OLD]+row[RECON]
    return original_check_graphs(logical,tokens)
def snapshots(report,read):
    assert report['tokens'] in (32,128) and len(report['prompt_ids'])==3968 and not report['performance_admitted']
    check_graphs(report['graphs'],report['tokens']);assert [r['prefix'] for r in report['snapshots']]==list(PREFIXES)
    for row in report['snapshots']:assert row['slots']==[row['prefix']] and set(row['caches'])=={'key','value'}
    return {(r['prefix'],name):read_cache(r['caches'][name],read) for r in report['snapshots'] for name in ('key','value')}
def compare(baseline,native,read_base,read_native):
    assert baseline['prompt_ids']==native['prompt_ids'] and baseline['tokens']==32 and native['tokens']==128
    a=snapshots(baseline,read_base);b=snapshots(native,read_native)
    rows=[dict(prefix=p,caches={n:cache_difference(b[(p,n)],a[(p,n)],p) for n in ('key','value')}) for p in PREFIXES]
    return dict(status='actual_KV128_layer3_complete_prefix_caches_replayed',prefixes=rows,all_prefix_KEY_and_VALUE_byte_exact=all(c['exact'] for r in rows for c in r['caches'].values()),first_sampled_different_prefix=next((r['prefix'] for r in rows if any(not c['exact'] for c in r['caches'].values())),None),complete_BF16_values_compared=6*2*4194304,full_model_MTP_quality_or_performance_qualified=False)
