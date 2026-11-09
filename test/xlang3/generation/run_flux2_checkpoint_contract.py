# SPDX-License-Identifier: Apache-2.0
"""Match the production FLUX graph's weight contract to an actual checkpoint.

Reads safetensors metadata only. Successful capture and matching weight shapes
do not prove numerical pretrained transformer inference.
"""
import json
import os
import struct
import sys
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..")))
import tensor
from xModel.flux2.klein_4b import transformer as flux

checkpoint=os.path.abspath(sys.argv[1])
with open(os.path.join(checkpoint,"config.json"),"r") as source: config=json.load(source)
with open(os.path.join(checkpoint,"diffusion_pytorch_model.safetensors"),"rb") as source:
    length=struct.unpack("<Q",source.read(8))[0]
    assert 0<length<16777216,length
    metadata=json.loads(source.read(length))
expected={}
heads=config["num_attention_heads"]
head_dim=config["attention_head_dim"]
hidden=heads*head_dim
intermediate=int(hidden*config["mlp_ratio"])
def linear(name,rows,columns): expected[name+".weight"]=[rows,columns]
linear("time_guidance_embed.timestep_embedder.linear_1",hidden,config["timestep_guidance_channels"])
linear("time_guidance_embed.timestep_embedder.linear_2",hidden,hidden)
for stream,sets in [("double_stream_modulation_img",6),("double_stream_modulation_txt",6),("single_stream_modulation",3)]:
    linear(stream+".linear",hidden*sets,hidden)
linear("x_embedder",hidden,config["in_channels"])
linear("context_embedder",hidden,config["joint_attention_dim"])
for index in range(config["num_layers"]):
    p="transformer_blocks."+str(index)
    for projection in ("to_q","to_k","to_v","add_q_proj","add_k_proj","add_v_proj","to_out.0","to_add_out"):
        linear(p+".attn."+projection,hidden,hidden)
    for name in ("norm_q","norm_k","norm_added_q","norm_added_k"):
        expected[p+".attn."+name+".weight"]=[head_dim]
    for name in ("ff","ff_context"):
        linear(p+"."+name+".linear_in",2*intermediate,hidden)
        linear(p+"."+name+".linear_out",hidden,intermediate)
for index in range(config["num_single_layers"]):
    p="single_transformer_blocks."+str(index)+".attn"
    linear(p+".to_qkv_mlp_proj",3*hidden+2*intermediate,hidden)
    linear(p+".to_out",hidden,hidden+intermediate)
    for name in ("norm_q","norm_k"): expected[p+"."+name+".weight"]=[head_dim]
linear("norm_out.linear",2*hidden,hidden)
linear("proj_out",config.get("out_channels") or config["in_channels"],hidden)
for name,shape in expected.items():
    assert name in metadata,("missing weight",name)
    assert metadata[name]["shape"]==shape,(name,shape,metadata[name])
    assert metadata[name]["dtype"] in ("BF16","F16","F32"),(name,metadata[name])

inputs=[tensor.input(name,shape=shape,dtype=tensor.float32) for name,shape in [
    ("image",[1,4,config["in_channels"]]),("text",[1,3,config["joint_attention_dim"]]),
    ("time",[1]),("cosine",[1,7,1,head_dim]),("sine",[1,7,1,head_dim])]]
dtype=tensor.bfloat16 if metadata["x_embedder.weight"]["dtype"]=="BF16" else tensor.float32
representative={"x_embedder.weight":tensor.input("x_embedder.weight",shape=metadata["x_embedder.weight"]["shape"],dtype=dtype)}
nodes=flux.Flux2Transformer(*inputs,representative,config).inspect()
referenced=set()
for node in nodes:
    attributes=node["attributes"]
    for key in ("weight_name","bias_name"):
        if key in attributes: referenced.add(attributes[key])
assert referenced==set(expected),(referenced-set(expected),set(expected)-referenced)
print("FLUX.2 production checkpoint contract passed",len(expected),"weight shapes;",
    config["num_layers"],"double and",config["num_single_layers"],"single blocks; pretrained transformer inference unvalidated",flush=True)
