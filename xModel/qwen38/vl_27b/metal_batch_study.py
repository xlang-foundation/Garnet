# SPDX-License-Identifier: Apache-2.0
"""Explicit unqualified batch2 text profile, separate from normal admission."""
def create_decoder(checkpoint,config,capacity):
    from .model import validate_config
    validate_config(config)
    q=config.get('quantization',{});text=config['text_config']
    if q.get('bits')!=4 or q.get('group_size')!=64 or q.get('mode')!='affine':
        raise ValueError('Batch study requires the pinned affine INT4/group64 artifact')
    if text.get('hidden_act')!='silu' or text.get('attention_bias') is not False:
        raise ValueError('Batch study requires the verified text projection contract')
    if type(capacity) is not int or not 64<=capacity<=4096:
        raise ValueError('Batch2 study requires a bounded64..4096 context')
    import garnet
    required=('mlx_qwen35_affine_int4_g64_bf16','native_continuous_scheduler',
              'bounded_prefill_workspace','retired_inplace_defaults','shared_prefill_scratch',
              'compact_attention_index','linear_chunk_conv','linear_chunk_delta')
    if any(tag not in garnet.backend_info() for tag in required):
        raise ValueError('Batch2 study requires guarded native scheduler, compressed weights and shared workspace extensions')
    from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
    decoder=MetalDecoder(checkpoint,text,2,capacity,required_checkpoint_format='mlx_qwen35_affine_int4_g64_bf16')
    try:decoder.enable_prefill(8,'shared',True)
    except:decoder.close();raise
    return decoder
