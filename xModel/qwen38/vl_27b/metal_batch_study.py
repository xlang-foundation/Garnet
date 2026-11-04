# SPDX-License-Identifier: Apache-2.0
"""Explicit unqualified batch2/4 text profiles, separate from normal admission."""
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder


class _ResidentBatchDecoder(MetalDecoder):
    def _supported_batches(self):return (2,4)

    def _wide_prefill_batches(self):return (2,)

    def _prefill_profiles(self):
        import garnet
        profiles=MetalDecoder._prefill_profiles(self)
        if self.batch==2 and 'qwen38_large_prefill' in garnet.backend_info():return profiles+(128,256)
        return profiles

    def __init__(self,*args,**kwargs):
        self._building_resident=True
        try:MetalDecoder.__init__(self,*args,**kwargs)
        finally:self._building_resident=False

    def reset(self):
        if self._building_resident and self.session is None:
            # The session constructs independent zeroed native state next.
            # No legacy host-state arrays are consumed on this resident route.
            self.poisoned=False;self.slots=[0 for _ in range(self.batch)]
            for layer in self.layers:layer['state1']=None;layer['state2']=None
        else:MetalDecoder.reset(self)


def create_decoder(checkpoint,config,capacity,batch=2,chunk=8):
    from .model import validate_config
    validate_config(config)
    q=config.get('quantization',{});text=config['text_config']
    if q.get('bits')!=4 or q.get('group_size')!=64 or q.get('mode')!='affine':
        raise ValueError('Batch study requires the pinned affine INT4/group64 artifact')
    if text.get('hidden_act')!='silu' or text.get('attention_bias') is not False:
        raise ValueError('Batch study requires the verified text projection contract')
    import garnet
    choices=(2,4,8,16,32,64)
    if 'qwen38_large_prefill' in garnet.backend_info():choices+=(128,256)
    if (type(capacity) is not int or not 64<=capacity<=4096 or type(batch) is not int or
        batch not in (2,4) or type(chunk) is not int or chunk not in (choices if batch==2 else (2,4,8))):
        raise ValueError('Batch study requires context64..4096 and an explicit native batch/chunk extension')
    if chunk>64 and any(tag not in garnet.backend_info() for tag in
        ('qwen38_large_prefill','attention_index_i16','retired_nonresident_packs')):
        raise ValueError('Large batch2 study requires 16-bit attention indices and retired nonresident packs')
    required=('mlx_qwen35_affine_int4_g64_bf16','native_continuous_scheduler',
              'bounded_prefill_workspace','retired_inplace_defaults','shared_prefill_scratch',
              'compact_attention_index','linear_chunk_conv','linear_chunk_delta')
    if any(tag not in garnet.backend_info() for tag in required):
        raise ValueError('Batch study requires guarded native scheduler, compressed weights and shared workspace extensions')
    if chunk>8 and any(tag not in garnet.backend_info() for tag in ('qwen38_wide_prefill','chunk_kv_inplace','row_final_hidden')):
        raise ValueError('Wide batch2 study requires bounded wide prefill, in-place cache and per-row hidden extensions')
    if batch==4 and 'chunk_kv_inplace' not in garnet.backend_info():
        raise ValueError('Batch4 study requires the in-place chunk-cache memory extension')
    decoder=_ResidentBatchDecoder(checkpoint,text,batch,capacity,required_checkpoint_format='mlx_qwen35_affine_int4_g64_bf16')
    try:decoder.enable_prefill(chunk,'shared',True)
    except:decoder.close();raise
    return decoder
