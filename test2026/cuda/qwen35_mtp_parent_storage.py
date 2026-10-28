"""Unwired construction-time parent-pool selection for a later MTP experiment.

Call before ANY MTP graph workspace/output binding or execution. The decoder
must outlive the MTP graphs. The caller must also reject MTP execution during a
pending target transaction, consume target packed outputs before MTP writes,
and validate full native lifecycle/state/output/sanitizer/memory/latency gates.
These CPU preflight checks alone establish no GPU alias or memory saving proof.
"""
import json


def select_parent_storage(decoder, programs, batch, hidden_size, cache_width):
    """Return existing roots plus exact accounting; perform no GPU allocation/bind."""
    if getattr(decoder, 'state_transaction', None) is not None:
        raise ValueError('Cannot select MTP storage during a target transaction')
    if not programs or set(programs) != {1, 4, 16} or batch not in (1, 2, 4, 8):
        raise ValueError('Parent storage experiment requires MTP1/4/16 and bounded batch')
    if hidden_size < 1 or cache_width < 1:
        raise ValueError('Invalid MTP output widths')
    if not getattr(decoder, 'shared_activation_storage', False) or not getattr(decoder, 'shared_output_storage', False):
        raise ValueError('Parent decoder must already own shared workspace and output pools')
    if not getattr(decoder, 'serial_target_graph_reuse', False) or not getattr(decoder, 'serial_graph_reuse', False):
        raise ValueError('Parent storage experiment requires serial target and MTP reuse')
    activation = getattr(decoder, 'activation_pool', None)
    output = getattr(decoder, 'output_pool', None)
    if activation is None or output is None:
        raise ValueError('Parent storage is unavailable')
    stats = [json.loads(program.graph.stats_json()) for program in programs.values()]
    if any(not s['user_managed_workspace_requested'] or s['external_workspace_bound']
           or s['external_output_storage'] or s['device_output_bytes'] != 0
           or s['aux_streams'] != 0 or not s['serial_graph_reuse_requested']
           or not s['cuda_graph_requested'] or s['graph_captures'] != 0
           or s['graph_replays'] != 0 for s in stats):
        raise ValueError('Select parent roots before the one-time bindings and graph execution')
    workspace_required = max(s['context_workspace_required_bytes'] for s in stats)
    output_required = batch * (max(programs) * hidden_size + 2 * cache_width) * 4
    activation_bytes = decoder.io.tensor_bytes(activation)
    output_bytes = decoder.io.tensor_bytes(output)
    if activation_bytes < workspace_required or output_bytes < output_required:
        raise ValueError('Parent pool capacity is smaller than an MTP graph requirement')
    return {'workspace': activation, 'output': output,
            'workspace_required_bytes': workspace_required, 'output_required_bytes': output_required,
            'borrowed_workspace_capacity_bytes': activation_bytes, 'borrowed_output_capacity_bytes': output_bytes,
            'new_owned_pool_bytes': 0,
            'accounting': 'Parent capacities remain owned by decoder; neither is added to session owned bytes.'}
