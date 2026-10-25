"""Admission for the experimental joined NVFP4 full decoder; no GPU imports."""
def validate_joined_mlp_profile(enabled,batch,capacity,chunk_size,external_compressed_weights,embedded_projection_weights=False,join_input_projections=False,*,bf16_cache_profile=False,bf16_float_reference=False):
    if type(bf16_float_reference) is not bool or (bf16_float_reference and bf16_cache_profile):raise ValueError('Float reference and packed BF16 joined profiles are exclusive booleans')
    if type(bf16_cache_profile) is not bool:raise ValueError('BF16 joined MLP profile requires an explicit boolean')
    if type(enabled) is not bool:
        raise ValueError('Joined MLP opt-in must be a boolean')
    if not enabled:
        if bf16_cache_profile or bf16_float_reference:raise ValueError('BF16 cache profile requires joined MLP projections')
        return
    typed=type(batch) is int and type(capacity) is int and type(chunk_size) is int
    original=typed and batch==1 and 32<=capacity<=4096 and chunk_size in (4,128)
    batch4_study=typed and batch==4 and capacity==4096 and chunk_size==4
    bf16_study=bf16_cache_profile and typed and batch in (4,8) and capacity==4096 and chunk_size in (4,32)
    float_reference=bf16_float_reference and typed and batch==4 and capacity==4096 and chunk_size in (4,32)
    if bf16_float_reference and not float_reference:raise ValueError('BF16 float reference joined projections require B4/cap4096/T4-or-T32')
    if not (original or batch4_study or bf16_study or float_reference):
        raise ValueError('Joined MLP study requires B1/cap32..4096/T4-or-T128 or experimental B4/cap4096/T4')
    if external_compressed_weights is not True or embedded_projection_weights or join_input_projections:
        raise ValueError('Joined MLP requires only the shared compressed pool')
