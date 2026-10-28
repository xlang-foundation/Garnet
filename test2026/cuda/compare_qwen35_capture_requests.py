"""Validate DOT-only capture requests without inventing Nsight metadata."""
import hashlib,json,sys
from pathlib import Path
source,control,output=map(Path,sys.argv[1:]);assert not output.exists()
rows,refs=[json.loads(p.read_text()) for p in (source,control)]
assert len(rows)==len(refs)==28
for index,(row,ref) in enumerate(zip(rows,refs)):
    for field in ('trial','workload_group','input_sha256','prompt_token_ids','generated_token_ids',
                  'sampled_tokens_including_stop','finish_reasons','stop_token_ids','profile','cache_policy',
                  'speculative_metrics','native_build_policy','visual_tokens_per_row'):
        assert row[field]==ref[field],(index,field)
    assert row['owned_memory']['owned_model_bytes']==ref['owned_memory']['owned_model_bytes']
    assert row['owned_memory']['speculative']['mtp_graph']['contexts']==ref['owned_memory']['speculative']['mtp_graph']['contexts']==1
    metadata=row['capture_topology_diagnostic']
    assert metadata['request_index']==index and not metadata['native_layer_profiler_enabled']
    assert not metadata['changed_existing_dot_files_this_request'] and not metadata['primary_latency_accepted']
result={'same_all28_input_output_stop_profile_speculative_counters':True,'all28_owned_bytes_and_mtp_context1_preserved':True,
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,control)},
        'scope':'DOT-only native capture metadata and exact original controls. No Nsight capture or primary latency/memory/quality acceptance.'}
with output.open('x') as stream:json.dump(result,stream,indent=2)
print('CAPTURE_REQUEST_PRESERVATION_PASS',len(rows))
