# SPDX-License-Identifier: Apache-2.0
"""Qwen3 causal prompt hidden states, reusable by language and image models.

Output layer indices follow Transformers: 0 is the token embedding, k is the
output of decoder layer k-1. Unrequested trailing layers do not affect these
hidden states and are absent from the captured graph.
"""
# garnet-dependency: ../generation/tensor_ops.py
from ..generation import tensor_ops as O
T=O.T

def round_hidden(value, config):
    return O.round_bfloat16(value) if config.get("dtype") == "bfloat16" else value

def norm(value, prefix, config):
    return round_hidden(O.rms_norm(value,prefix,config["rms_norm_eps"],
        round_before_scale=config.get("dtype")=="bfloat16"),config)

@T.fusion(role="qwen3_prompt_layer",atomic=True)
def layer(value, cosine, sine, mask, batch, tokens, config, index):
    heads=config["num_attention_heads"]
    kv_heads=config["num_key_value_heads"]
    width=config["head_dim"]
    p="model.layers."+str(index)
    normalized=norm(value,p+".input_layernorm",config)
    query=O.reshape(O.linear(normalized,p+".self_attn.q_proj"),[batch,tokens,heads,width])
    key=O.reshape(O.linear(normalized,p+".self_attn.k_proj"),[batch,tokens,kv_heads,width])
    values=O.reshape(O.linear(normalized,p+".self_attn.v_proj"),[batch,tokens,kv_heads,width])
    query=norm(query,p+".self_attn.q_norm",config)
    key=norm(key,p+".self_attn.k_norm",config)
    query=round_hidden(O.rotary_split_half(query,cosine,sine),config)
    key=round_hidden(O.rotary_split_half(key,cosine,sine),config)
    attended=round_hidden(O.attention(query,key,values,mask),config)
    attended=O.reshape(attended,[batch,tokens,heads*width])
    value=round_hidden(value+O.linear(attended,p+".self_attn.o_proj"),config)
    normalized=norm(value,p+".post_attention_layernorm",config)
    gate=round_hidden(O.silu(O.linear(normalized,p+".mlp.gate_proj")),config)
    up=O.linear(normalized,p+".mlp.up_proj")
    feedforward=O.linear(round_hidden(gate*up,config),p+".mlp.down_proj")
    return round_hidden(value+feedforward,config)

def hidden_states(input_ids, cosine, sine, mask, config, output_layers):
    if config.get("model_type") != "qwen3" or config.get("attention_bias",False):
        raise ValueError("Qwen3 hidden-state program requires the bias-free Qwen3 architecture")
    if config.get("use_sliding_window",False) or config.get("rope_scaling"):
        raise ValueError("Qwen3 prompt encoder currently requires full attention and unscaled RoPE")
    if config.get("hidden_act","silu") != "silu":
        raise ValueError("Qwen3 prompt encoder requires SiLU")
    if not output_layers or sorted(set(output_layers)) != list(output_layers) or output_layers[0]<1 or output_layers[-1]>config["num_hidden_layers"]:
        raise ValueError("hidden-state layer indices must be increasing, unique, and inside the checkpoint")
    batch,tokens=input_ids.shape
    value=O.cast(O.embedding(input_ids,"model.embed_tokens"),"float32")
    cosine=round_hidden(cosine,config)
    sine=round_hidden(sine,config)
    outputs=[]
    for index in range(output_layers[-1]):
        value=layer(value,cosine,sine,mask,batch,tokens,config,index)
        if index+1 in output_layers: outputs.append(value)
    output=outputs[0]
    for value in outputs[1:]: output=O.concat(output,value,-1)
    return output
