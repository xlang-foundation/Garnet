"""Audit all real requests, including discrete speculation work, across joining."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('candidate',type=Path);p.add_argument('output',type=Path)
p.add_argument('--linear-layers',type=int,required=True)
feature=p.add_mutually_exclusive_group();feature.add_argument('--activation-storage-transition',action='store_true');feature.add_argument('--internal-packed-store-transition',action='store_true');a=p.parse_args()
assert not a.output.exists(),'Preserve workload evidence'
load=lambda path:{(row['trial'],row['workload_group']):row for row in json.loads(path.read_text(encoding='utf-8'))}
control=load(a.control);candidate=load(a.candidate)
assert len(control)==len(candidate)==28 and control.keys()==candidate.keys()
fields=('input_sha256','prompt_token_ids','generated_token_ids','sampled_tokens_including_stop',
    'finish_reasons','stop_token_ids','cache_policy','speculative','speculative_metrics','warmup')
differences=[]
for key,left in control.items():
    right=candidate[key]
    for field in fields:
        if left[field]!=right[field]:differences.append({'trial':key[0],'group':key[1],'field':field})
    lp=dict(left['profile']);rp=dict(right['profile'])
    transition='internalPackedStores' if a.internal_packed_store_transition else 'sharePackedActivationStorage' if a.activation_storage_transition else 'joinDecodeInputProjections'
    if lp.pop(transition,False) or rp.pop(transition,False) is not True:
        differences.append({'trial':key[0],'group':key[1],'field':transition+' transition'})
    if lp!=rp:differences.append({'trial':key[0],'group':key[1],'field':'other profile fields'})
    native=right['owned_memory']['decoder']
    if (right['io_memory']['joined_projection_group_count']!=a.linear_layers or
            sum(graph['joined_projection_inputs'] for graph in native['graphs'])!=a.linear_layers):
        differences.append({'trial':key[0],'group':key[1],'field':'native joining counters'})
    if a.activation_storage_transition or a.internal_packed_store_transition:
        if not (native['shared_activation_pool_bytes']>0 and all(g['user_managed_workspace'] and g['external_workspace_bound'] and g['external_workspace_capacity_bytes']==native['shared_activation_pool_bytes'] for g in native['graphs'][:-1])):
            differences.append({'trial':key[0],'group':key[1],'field':'native activation storage counters'})
        if a.internal_packed_store_transition:
            bodies=native['graphs'][:-1];prior=left['owned_memory']['decoder']
            # Both models have one full-attention layer per three linear layers.
            layers=a.linear_layers+a.linear_layers//3
            if sum(g['internal_packed_stores'] for g in bodies)!=len(bodies)*layers or sum(g['internal_packed_root_fills'] for g in bodies)!=len(bodies) or any(g['network_output_aliases_input'] for g in native['graphs']):
                differences.append({'trial':key[0],'group':key[1],'field':'native internal store counters'})
            if any(g['internal_packed_stores'] or g['internal_packed_root_fills'] for g in prior['graphs']):
                differences.append({'trial':key[0],'group':key[1],'field':'control unexpectedly uses internal stores'})
            for field in ('state_bytes','request_buffer_bytes','engine_output_bytes','shared_output_pool_bytes',
                          'embedding_head_shared_bytes','head_input_bytes','head_output_bytes','proposal_io_bytes',
                          'projection_weight_pool_bytes','projection_weight_pool_entries','joined_projection_group_count','joined_projection_group_bytes'):
                if prior[field]!=native[field]:differences.append({'trial':key[0],'group':key[1],'field':'changed owner '+field})
    elif left['owned_memory']['owned_model_bytes']!=right['owned_memory']['owned_model_bytes']:
        differences.append({'trial':key[0],'group':key[1],'field':'retained allocation'})
result={'sources':[{'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in (a.control,a.candidate)],
    'all_28_checked':True,'checked_fields':fields,'differences':differences,'all_passed':not differences,
    'activation_storage_transition':a.activation_storage_transition,
    'internal_packed_store_transition':a.internal_packed_store_transition,
    'retained_allocation_bytes':{'control':sorted({row['owned_memory']['owned_model_bytes'] for row in control.values()}),'candidate':sorted({row['owned_memory']['owned_model_bytes'] for row in candidate.values()})},
    'scope':'All-request inputs, outputs, stops, cache declarations, discrete MTP calls/acceptance/iterations, native joining counters and retained allocation. This is workload preservation, not independent quality, full-state equivalence for every request, timing significance, physical memory or a cross-engine precision equivalence claim.'}
a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'all_passed':result['all_passed'],'differences':differences},indent=2))
assert result['all_passed'],'Joined workload changed; preserve differences'
