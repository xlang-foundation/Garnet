"""Admission for the experimental joined NVFP4 full decoder; no GPU imports."""
def validate_joined_mlp_profile(enabled,batch,capacity,chunk_size,external_compressed_weights,embedded_projection_weights=False,join_input_projections=False):
    if type(enabled) is not bool:
        raise ValueError('Joined MLP opt-in must be a boolean')
    if not enabled:
        return
    typed=type(batch) is int and type(capacity) is int and type(chunk_size) is int
    original=typed and batch==1 and 32<=capacity<=4096 and chunk_size in (4,128)
    batch4_study=typed and batch==4 and capacity==4096 and chunk_size==4
    if not (original or batch4_study):
        raise ValueError('Joined MLP study requires B1/cap32..4096/T4-or-T128 or experimental B4/cap4096/T4')
    if external_compressed_weights is not True or embedded_projection_weights or join_input_projections:
        raise ValueError('Joined MLP requires only the shared compressed pool')
