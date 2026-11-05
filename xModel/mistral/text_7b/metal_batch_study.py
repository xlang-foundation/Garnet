# SPDX-License-Identifier: Apache-2.0
"""Unqualified native continuous-batch Mistral text profile factory."""
import garnet
from .metal_decoder import MistralMetalDecoder


def create_decoder(checkpoint,config,capacity,batch=4,chunk=128,compact=False,maximum_owner_chunk=None):
    backend=garnet.backend_info()
    required=('prepared_dense_prefill','native_request_row_leases','native_continuous_scheduler',
              'chunked_prefill_scheduler','row_final_hidden','compact_attention_index',
              'chunk_kv_inplace','retired_inplace_defaults','shared_prefill_scratch',
              'affine_coefficient_dtype=fp16','bounded_affine_kernel_profiles')
    if any(tag not in backend for tag in required):
        raise ValueError('Mistral batch study requires native dense profiles, row leases, scheduler and private KV')
    if type(compact) is not bool or (compact and (batch!=4 or 'compact_owner_prefill' not in backend)):
        raise ValueError('Compact owner study requires the explicit native B4 family extension')
    if maximum_owner_chunk is not None and (not compact or type(maximum_owner_chunk) is not int or
        maximum_owner_chunk not in (2,4,8,16,32,64,128,256,512) or maximum_owner_chunk>capacity or
        maximum_owner_chunk<chunk or (maximum_owner_chunk>128 and 'wide_owner_prefill' not in backend)):
        raise ValueError('Owner-prefill maximum requires an explicit compiled compact profile')
    limits=[int(field.strip().split('=')[1]) for field in backend.split(';')
            if field.strip().startswith('dense_prefill_max_tokens=')]
    if len(limits)!=1 or limits[0] not in (16,32,64,128,256,512):
        raise ValueError('Mistral batch study requires one bounded dense token-row profile')
    if (type(batch) is not int or batch not in (1,2,4) or type(capacity) is not int or
        not 64<=capacity<=4096 or type(chunk) is not int or
        chunk not in (2,4,8,16,32,64,128,256,512) or chunk>capacity or batch*chunk>limits[0]):
        raise ValueError('Mistral batch/context/chunk exceeds its compiled global token-row profile')
    decoder=MistralMetalDecoder(checkpoint,config,batch,capacity,resident=True)
    try:
        decoder.prepare_prefill(chunk)
        if compact:decoder.prepare_compact_prefill(chunk if maximum_owner_chunk is None else maximum_owner_chunk)
    except:decoder.close();raise
    return decoder
