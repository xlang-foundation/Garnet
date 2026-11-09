# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 Klein distilled transformer expressed as portable Tensor Expressions.

Checkpoint naming follows the official Diffusers-layout Klein 4B release.
Image tokens include reference-image tokens when editing; text tokens always
precede image tokens in joint attention. This program uses full joint attention,
not the separate KV-cached variant of Klein.
"""
# garnet-dependency: ../../generation/tensor_ops.py
from ...generation import tensor_ops as O
import tensor
T = O.T

GARNET_MODEL_SPEC = {"arguments": [
    {"name":"image", "kind":"tensor"},
    {"name":"text", "kind":"tensor"},
    {"name":"timestep", "kind":"tensor"},
    {"name":"cosine", "kind":"tensor"},
    {"name":"sine", "kind":"tensor"},
    {"name":"weights", "kind":"weights"},
    {"name":"config", "kind":"config"},
]}

def validate_config(config):
    if config.get("guidance_embeds", False):
        raise ValueError("Klein distilled program requires guidance_embeds=false")
    if config.get("patch_size",1) != 1:
        raise ValueError("FLUX.2 transformer expects already packed latent tokens")
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    if heads < 1 or head_dim < 2 or head_dim % 2:
        raise ValueError("FLUX.2 attention dimensions invalid")
    if sum(config["axes_dims_rope"]) != head_dim or any(d<2 or d%2 for d in config["axes_dims_rope"]):
        raise ValueError("FLUX.2 rotary axes must partition the head dimension into even widths")
    if config["num_layers"] < 0 or config["num_single_layers"] < 0:
        raise ValueError("FLUX.2 layer counts invalid")
    if config["num_layers"] + config["num_single_layers"] < 1:
        raise ValueError("FLUX.2 transformer requires at least one block")
    if config.get("eps",0.000001) <= 0:
        raise ValueError("FLUX.2 normalization epsilon must be positive")

def modulation_parts(value, batch, hidden, count):
    return [O.reshape(O.slice(value,-1,index*hidden,hidden),[batch,1,hidden]) for index in range(count)]

def rounded(value,config):
    return O.round_bfloat16(value) if config.get("_bf16_math",False) else value

def modulated_norm(value,shift,scale,config):
    value=rounded(O.layer_norm(value,config.get("eps",0.000001)),config)
    if config.get("_bf16_math",False):
        return rounded(rounded(value*rounded(scale+1.0,config),config)+shift,config)
    return O.modulate(value,shift,scale)

def gated_swiglu(value,width,config):
    if config.get("_bf16_math",False):
        gate=rounded(O.silu(O.slice(value,-1,0,width)),config)
        return rounded(gate*O.slice(value,-1,width,width),config)
    return O.swiglu(value)

def feed_forward(value, prefix,config):
    width=int(config["num_attention_heads"]*config["attention_head_dim"]*config.get("mlp_ratio",3.0))
    return O.linear(gated_swiglu(O.linear(value,prefix+".linear_in"),width,config),prefix+".linear_out")

def stream_projections(value, prefix, context, batch, tokens, config):
    heads = config["num_attention_heads"]
    width = config["attention_head_dim"]
    eps = config.get("eps",0.000001)
    names = ("add_q_proj","add_k_proj","add_v_proj") if context else ("to_q","to_k","to_v")
    query,key,values = [O.reshape(O.linear(value,prefix+"."+name),[batch,tokens,heads,width]) for name in names]
    query = rounded(O.rms_norm(query,prefix+(".norm_added_q" if context else ".norm_q"),eps),config)
    key = rounded(O.rms_norm(key,prefix+(".norm_added_k" if context else ".norm_k"),eps),config)
    return query,key,values

@T.fusion(role="flux2_double_stream_block", atomic=True)
def double_block(image, text, image_mod, text_mod, cosine, sine, batch, image_tokens, text_tokens, config, index):
    hidden = config["num_attention_heads"]*config["attention_head_dim"]
    eps = config.get("eps",0.000001)
    prefix = "transformer_blocks."+str(index)
    im = modulation_parts(image_mod,batch,hidden,6)
    tm = modulation_parts(text_mod,batch,hidden,6)
    normalized_image = modulated_norm(image,im[0],im[1],config)
    normalized_text = modulated_norm(text,tm[0],tm[1],config)
    iq,ik,iv = stream_projections(normalized_image,prefix+".attn",False,batch,image_tokens,config)
    tq,tk,tv = stream_projections(normalized_text,prefix+".attn",True,batch,text_tokens,config)
    query = rounded(O.rotary(O.concat(tq,iq,1),cosine,sine),config)
    key = rounded(O.rotary(O.concat(tk,ik,1),cosine,sine),config)
    values = O.concat(tv,iv,1)
    attended = O.reshape(rounded(O.attention(query,key,values,compute_dtype=config.get('_attention_dtype')),config),[batch,text_tokens+image_tokens,hidden])
    image = rounded(image+rounded(im[2]*O.linear(O.slice(attended,1,text_tokens,image_tokens),prefix+".attn.to_out.0"),config),config)
    text = rounded(text+rounded(tm[2]*O.linear(O.slice(attended,1,0,text_tokens),prefix+".attn.to_add_out"),config),config)
    image = rounded(image+rounded(im[5]*feed_forward(modulated_norm(image,im[3],im[4],config),prefix+".ff",config),config),config)
    text = rounded(text+rounded(tm[5]*feed_forward(modulated_norm(text,tm[3],tm[4],config),prefix+".ff_context",config),config),config)
    return image,text

@T.fusion(role="flux2_single_stream_block", atomic=True)
def single_block(value, modulation, cosine, sine, batch, tokens, config, index):
    heads = config["num_attention_heads"]
    head_dim = config["attention_head_dim"]
    hidden = heads*head_dim
    intermediate = int(hidden*config.get("mlp_ratio",3.0))
    eps = config.get("eps",0.000001)
    prefix = "single_transformer_blocks."+str(index)+".attn"
    shift,scale,gate = modulation_parts(modulation,batch,hidden,3)
    normalized = modulated_norm(value,shift,scale,config)
    packed = O.linear(normalized,prefix+".to_qkv_mlp_proj")
    query,key,values = [O.reshape(O.slice(packed,-1,part*hidden,hidden),[batch,tokens,heads,head_dim]) for part in range(3)]
    query = rounded(O.rotary(rounded(O.rms_norm(query,prefix+".norm_q",eps),config),cosine,sine),config)
    key = rounded(O.rotary(rounded(O.rms_norm(key,prefix+".norm_k",eps),config),cosine,sine),config)
    attended = O.reshape(rounded(O.attention(query,key,values,compute_dtype=config.get('_attention_dtype')),config),[batch,tokens,hidden])
    feedforward = gated_swiglu(O.slice(packed,-1,3*hidden,2*intermediate),intermediate,config)
    return rounded(value+rounded(gate*O.linear(O.concat(attended,feedforward,-1),prefix+".to_out"),config),config)

@T.fusion(name="flux2_klein_transformer", boundary="required")
def Flux2Transformer(image, text, timestep, cosine, sine, weights, config):
    validate_config(config)
    config=dict(config)
    representative=weights.get("x_embedder.weight")
    config["_bf16_math"]=representative is not None and representative.dtype==tensor.bfloat16
    batch,image_tokens,_ = config.get("_image_shape",image.shape)
    text_batch,text_tokens,_ = text.shape
    if batch != text_batch:
        raise ValueError("FLUX.2 text and image batches differ")
    hidden = config["num_attention_heads"]*config["attention_head_dim"]
    image=rounded(image,config)
    text=rounded(text,config)
    time_input=rounded(rounded(timestep,config)*1000.0,config)
    time = rounded(O.sinusoidal(time_input,config.get("timestep_guidance_channels",256)),config)
    time = O.linear(rounded(O.silu(O.linear(time,"time_guidance_embed.timestep_embedder.linear_1")),config),
        "time_guidance_embed.timestep_embedder.linear_2")
    activated_time = rounded(O.silu(time),config)
    image_mod = O.linear(activated_time,"double_stream_modulation_img.linear")
    text_mod = O.linear(activated_time,"double_stream_modulation_txt.linear")
    single_mod = O.linear(activated_time,"single_stream_modulation.linear")
    image = O.linear(image,"x_embedder")
    text = O.linear(text,"context_embedder")
    for index in range(config["num_layers"]):
        image,text = double_block(image,text,image_mod,text_mod,cosine,sine,batch,image_tokens,text_tokens,config,index)
    value = O.concat(text,image,1)
    for index in range(config["num_single_layers"]):
        value = single_block(value,single_mod,cosine,sine,batch,image_tokens+text_tokens,config,index)
    value = O.slice(value,1,text_tokens,image_tokens)
    output_mod = O.linear(activated_time,"norm_out.linear")
    scale,shift = modulation_parts(output_mod,batch,hidden,2)
    value = modulated_norm(value,shift,scale,config)
    return O.linear(value,"proj_out")


@T.fusion(name="flux2_klein_transformer_fused_attention", boundary="required")
def Flux2TransformerFusedAttention(image, text, timestep, cosine, sine, weights, config):
    options = dict(config)
    options['_attention_dtype'] = 'float16'
    return Flux2Transformer(image,text,timestep,cosine,sine,weights,options)
