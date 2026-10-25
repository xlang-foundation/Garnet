"""Correctness reference: BF16 values in full FP32 caches; never compact state."""
from .bf16_kv_profile import validate_bf16_kv_program,validate_bf16_kv_decoder


def validate_bf16_float_reference_program(enabled,batch,capacity,chunk,retain,config,*,bf16_kv_state=False,**flags):
    if type(enabled) is not bool:raise ValueError('BF16 float reference opt-in must be boolean')
    if not enabled:return
    if type(batch) is not int or batch!=4 or bf16_kv_state is not False:
        raise ValueError('BF16 float reference requires B4 and excludes packed BF16 state')
    validate_bf16_kv_program(True,batch,capacity,chunk,retain,config,**flags)


def validate_bf16_float_reference_decoder(enabled,batch,capacity,prefill,decode,config,*,bf16_kv_state=False,**flags):
    if type(enabled) is not bool:raise ValueError('BF16 float reference opt-in must be boolean')
    if not enabled:return
    if type(batch) is not int or batch!=4 or bf16_kv_state is not False:
        raise ValueError('BF16 float reference requires B4 and excludes packed BF16 state')
    validate_bf16_kv_decoder(True,batch,capacity,prefill,decode,config,**flags)
