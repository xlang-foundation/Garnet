# SPDX-License-Identifier: Apache-2.0
"""Construction-only shared roots for serialized, owner-guarded tree phases."""
import json

def bind_parent_storage(graph,parent,batch,output_width):
    if parent.state_transaction is not None or parent.poisoned or parent.head is None:
        raise ValueError('Select shared storage only with an idle healthy parent')
    if not (parent.shared_activation_storage and parent.shared_output_storage and parent.serial_target_graph_reuse and parent.serial_graph_reuse):
        raise ValueError('Tree storage requires explicit serial parent workspace/output pools')
    if parent.activation_pool is None or parent.output_pool is None or batch!=parent.batch or output_width<1:
        raise ValueError('Parent storage unavailable or output dimensions invalid')
    stats=json.loads(graph.stats_json())
    if not stats['user_managed_workspace_requested'] or stats['external_workspace_bound'] or stats['external_output_storage'] or stats['device_output_bytes'] or stats['aux_streams'] or not stats['serial_graph_reuse_requested'] or not stats['cuda_graph_requested'] or stats['graph_captures'] or stats['graph_replays']:
        raise ValueError('Shared binding requires an unexecuted, unbound serial managed graph')
    if parent.io.tensor_bytes(parent.activation_pool)<stats['context_workspace_required_bytes'] or parent.io.tensor_bytes(parent.output_pool)<batch*output_width*4:
        raise ValueError('Tree graph exceeds the existing parent storage capacity: workspace required='+str(stats['context_workspace_required_bytes'])+' available='+str(parent.io.tensor_bytes(parent.activation_pool))+'; output required='+str(batch*output_width*4)+' available='+str(parent.io.tensor_bytes(parent.output_pool)))
    # All validation precedes the one-time bindings. Tree borrows keep parent
    # alive; transaction phases forbid drafting after target packet publication.
    graph.set_workspace_storage(parent.activation_pool)
    graph.set_output_storage(parent.io.storage_view(parent.output_pool,[batch,output_width]))
