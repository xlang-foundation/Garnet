"""The layer2 observation uses ordinary operators; the layer0/1 diagnostic guard stays unchanged."""
from .gdn_chunk_stage_layout import layout,pieces
from .fp8_z128_stage_layout import check_graph
FIXTURE='evidence/layer2-ordinary-common-input.f32'
META='evidence/layer2-ordinary-common-input.json'
INPUT_SOURCE='retained_T32_layer1_ordinary_output_common_input'

def validate_profile(layer,tokens,diagnostic):
    if type(layer) is not int or layer!=2 or type(tokens) is not int or tokens not in (32,128) or diagnostic is not False:
        raise ValueError('Only ordinary layer2 B1 T32/T128 observation is permitted')
    return layout(tokens,False)
