"""CPU-only B1 prototype admission; no production/native/model admission changes."""

def _boolean(value,name):
    if type(value) is not bool:raise ValueError(name+' requires an explicit boolean')

def validate_bf16_low_batch_kv_program(enabled,batch,capacity,chunk,retain,config,*,external_compressed_weights,
                             joined_mlp_projections,embedded_projection_weights=False,legacy_bf16_kv_state=False,bf16_kv_float_reference=False,
                             join_input_projections=False,kv_delta_state=False,kv_overlay_strategy=1):
    _boolean(enabled,'BF16 KV opt-in')
    if not enabled:return
    if legacy_bf16_kv_state is not False or bf16_kv_float_reference is not False:
        raise ValueError('B1 packed BF16 excludes legacy B4/B8 and full FP32 carrier reference modes')
    if any(type(v) is not int for v in (batch,capacity,chunk,kv_overlay_strategy)) or batch!=1 or capacity!=4096 or chunk not in (4,32) or kv_overlay_strategy!=1:
        raise ValueError('BF16 KV program requires B1,cap4096,T4/T32 and fixed strategy1')
    if retain is not (chunk==4) or external_compressed_weights is not True or joined_mlp_projections is not True or embedded_projection_weights is not False or join_input_projections is not False or kv_delta_state is not False:
        raise ValueError('BF16 KV requires shared mixed joined projections, T4-only replay and excludes legacy/embedded/joined-input modes')
    wanted={'hidden_size':5120,'intermediate_size':17408,'num_attention_heads':24,'num_key_value_heads':4,'head_dim':256,
            'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_key_head_dim':128,'linear_value_head_dim':128,'linear_conv_kernel_dim':4}
    if any(type(config.get(k)) is not int or config[k]!=v for k,v in wanted.items()) or config.get('layer_types')!=['full_attention' if i%4==3 else 'linear_attention' for i in range(64)]:
        raise ValueError('BF16 KV requires the pinned27B 64-layer geometry')
    rope=config.get('rope_parameters',{})
    if rope.get('partial_rotary_factor')!=.25 or rope.get('mrope_section')!=[11,11,10] or rope.get('rope_theta')!=10000000 or config.get('rms_norm_eps')!=1e-6:
        raise ValueError('BF16 KV requires the pinned normalization/RoPE configuration')

def validate_bf16_low_batch_kv_decoder(enabled,batch,capacity,prefill,decode,config,*,retain_linear_inputs,
                            deferred_state_commit,shared_output_storage,host_embedding_lookup,**program_flags):
    _boolean(enabled,'BF16 KV opt-in')
    if not enabled:return
    if type(prefill) is not int or prefill!=32 or type(decode) is not int or decode!=4:
        raise ValueError('BF16 KV decoder requires prefillT32/decodeT4')
    if any(v is not True for v in (retain_linear_inputs,deferred_state_commit,shared_output_storage,host_embedding_lookup)):
        raise ValueError('BF16 KV decoder requires retained/deferred/shared-output/host-embedding mode')
    validate_bf16_low_batch_kv_program(True,batch,capacity,decode,True,config,**program_flags)
    validate_bf16_low_batch_kv_program(True,batch,capacity,prefill,False,config,**program_flags)
