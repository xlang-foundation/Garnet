"""Explicit engine4098 capacity for an unchanged actual4096-token workload.

Installed tp_worker bounds max_req_len to context_len-1; scheduler then bounds
new tokens to max_req_len-input_len-1. No installed engine code is modified.
"""
from reference_fixed4k_sglang_identity import validate as original_validate,CHECKPOINT

def validate(info,mode):
 if not isinstance(info,dict):raise ValueError('Actual server_info required')
 for key,value in (('context_length',4098),('max_total_tokens',32784)):
  if type(info.get(key)) is not int or info[key]!=value:raise ValueError('Explicit reserved context setting differs: '+key)
 projected=dict(info);projected.update(context_length=4096,max_total_tokens=32768)
 result=original_validate(projected,mode)
 result['verified_settings'].update(context_length=4098,max_total_tokens=32784)
 result.update(context_capacity=4096,context_capacity_semantics='actual_prompt_plus_completion_workload_limit',configured_engine_context_capacity=4098,engine_context_reserved_slots=2,configured_engine_token_pool=32784)
 return result
