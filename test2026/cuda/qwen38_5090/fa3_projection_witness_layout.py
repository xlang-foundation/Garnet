"""Embedded-compatible metadata; numerical replay stays in external Python."""
import hashlib,math
STAGES=(('normalized',5120),('key',1024),('value',1024))
sha=lambda b:hashlib.sha256(b).hexdigest()
def check_graph(stats,tokens):
    assert type(tokens) is int and tokens in (32,128) and stats['checkpoint_weight_precision']=='modelopt_mixed'
    for key,value in [('native_fp8_projection_weights',2),('native_nvfp4_projection_weights',0),('native_int4_g128_projection_weights',0),('external_compressed_weight_inputs',0),('external_bf16_weight_inputs',0),('native_fp8_m128_canonical_projections',0),('native_fp8_z128_canonical_projections',0),('native_fp8_z128_reconstructed_projections',0),('device_output_bytes',tokens*7168*4),('graph_capture_failures',0),('graph_captures',1)]:
        assert type(stats[key]) is int and stats[key]==value,(key,stats[key])
    assert type(stats['graph_replays']) is int and stats['graph_replays']>=1
def weight_metadata(rows):
    assert set(rows)=={'model.language_model.layers.3.self_attn.k_proj.weight','model.language_model.layers.3.self_attn.v_proj.weight'}
    for row in rows.values():
        for k,v in [('logical_shape',[1024,5120]),('packed_shape',[1024,1280]),('packed_dtype','INT32'),('packed_bytes',5242880),('recipe','modelopt_fp8_w8a8'),('scale_bytes',0),('scale_shape',[]),('joined_mlp',False)]:assert type(row[k]) is type(v) and row[k]==v,(k,row[k])
        for k in ('weight_scale','input_scale'):assert type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>0
