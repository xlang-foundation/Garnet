# SPDX-License-Identifier: Apache-2.0
"""Small independent scalar FLUX.2 reference; no tensor framework or CPython.

This deliberately uses row lists and scalar reductions rather than evaluating
the production Tensor Expressions. It checks wiring and weight conventions,
not pretrained quality or low-precision parity.
"""
import json
import math
import os
import struct

CONFIG = {"guidance_embeds":False,"num_attention_heads":2,"attention_head_dim":8,
    "axes_dims_rope":[2,2,2,2],"num_layers":1,"num_single_layers":1,
    "in_channels":4,"joint_attention_dim":12,"mlp_ratio":3.0,"eps":0.000001,
    "timestep_guidance_channels":8,"patch_size":1,"rope_theta":2000}

def fixture(directory):
    weights = {}
    shapes = {}
    def add(name, shape, norm=False):
        count = math.prod(shape)
        serial = len(weights)+1
        values = [(0.8+0.1*math.sin(index+serial)) if norm else
            0.08*math.sin(index*0.37+serial*0.53) for index in range(count)]
        values = list(struct.unpack("<"+"f"*count,struct.pack("<"+"f"*count,*values)))
        weights[name] = values
        shapes[name] = shape
    def linear(name, rows, columns): add(name+".weight",[rows,columns])
    linear("time_guidance_embed.timestep_embedder.linear_1",16,8)
    linear("time_guidance_embed.timestep_embedder.linear_2",16,16)
    for stream,sets in [("double_stream_modulation_img",6),("double_stream_modulation_txt",6),("single_stream_modulation",3)]:
        linear(stream+".linear",16*sets,16)
    linear("x_embedder",16,4)
    linear("context_embedder",16,12)
    p = "transformer_blocks.0"
    for projection in ("to_q","to_k","to_v","add_q_proj","add_k_proj","add_v_proj","to_out.0","to_add_out"):
        linear(p+".attn."+projection,16,16)
    for name in ("norm_q","norm_k","norm_added_q","norm_added_k"):
        add(p+".attn."+name+".weight",[8],True)
    for name in ("ff","ff_context"):
        linear(p+"."+name+".linear_in",96,16)
        linear(p+"."+name+".linear_out",16,48)
    p = "single_transformer_blocks.0.attn"
    linear(p+".to_qkv_mlp_proj",144,16)
    linear(p+".to_out",16,64)
    for name in ("norm_q","norm_k"): add(p+"."+name+".weight",[8],True)
    linear("norm_out.linear",32,16)
    linear("proj_out",4,16)
    payload = b""
    header = {}
    for name,values in weights.items():
        data = struct.pack("<"+"f"*len(values),*values)
        header[name] = {"dtype":"F32","shape":shapes[name],"data_offsets":[len(payload),len(payload)+len(data)]}
        payload += data
    encoded = json.dumps(header,separators=(",",":")).encode("utf-8")
    encoded += b" "*((-len(encoded))%8)
    os.makedirs(directory,exist_ok=True)
    contents = struct.pack("<Q",len(encoded))+encoded+payload
    checkpoint = os.path.join(directory,"model.safetensors")
    existing = b""
    if os.path.isfile(checkpoint):
        with open(checkpoint,"rb") as source: existing = source.read()
    if existing != contents:
        with open(checkpoint,"wb") as output: output.write(contents)
    config_path = os.path.join(directory,"config.json")
    contents = json.dumps(CONFIG)
    existing = ""
    if os.path.isfile(config_path):
        with open(config_path,"r") as source: existing = source.read()
    if existing != contents:
        with open(config_path,"w") as output: output.write(contents)
    return weights,shapes

def forward(weights, shapes, image, text, timestep, cosine, sine):
    def linear(rows, name):
        data = weights[name+".weight"]
        out_width,in_width = shapes[name+".weight"]
        return [[sum(row[k]*data[j*in_width+k] for k in range(in_width)) for j in range(out_width)] for row in rows]
    def silu(value): return value/(1+math.exp(-value))
    def norm(rows):
        result = []
        for row in rows:
            mean = sum(row)/len(row)
            variance = sum((x-mean)**2 for x in row)/len(row)
            result.append([(x-mean)/math.sqrt(variance+0.000001) for x in row])
        return result
    def modulate(rows, shift, scale):
        return [[x*(1+scale[index])+shift[index] for index,x in enumerate(row)] for row in rows]
    def residual(rows, update, gate):
        return [[x+gate[index]*y for index,(x,y) in enumerate(zip(a,b))] for a,b in zip(rows,update)]
    def swiglu(rows):
        half = len(rows[0])//2
        return [[silu(row[index])*row[index+half] for index in range(half)] for row in rows]
    def rms(rows, prefix):
        scale = weights[prefix+".weight"]
        result = []
        for row in rows:
            normalized = []
            for head in range(2):
                part = row[head*8:(head+1)*8]
                denominator = math.sqrt(sum(x*x for x in part)/8+0.000001)
                normalized += [part[index]/denominator*scale[index] for index in range(8)]
            result.append(normalized)
        return result
    def rope(rows):
        result = []
        for position,row in enumerate(rows):
            rotated = []
            for head in range(2):
                for pair in range(4):
                    channel = pair*2
                    a,b = row[head*8+channel:head*8+channel+2]
                    rotated += [a*cosine[position][channel]-b*sine[position][channel],
                        b*cosine[position][channel+1]+a*sine[position][channel+1]]
            result.append(rotated)
        return result
    def attention(q,k,v):
        result = []
        for query in q:
            output = []
            for head in range(2):
                start = head*8
                scores = [sum(query[start+i]*key[start+i] for i in range(8))/math.sqrt(8) for key in k]
                probability = [math.exp(x-max(scores)) for x in scores]
                denominator = sum(probability)
                output += [sum(probability[j]*v[j][start+i] for j in range(len(k)))/denominator for i in range(8)]
            result.append(output)
        return result
    def chunks(row, count): return [row[index*16:(index+1)*16] for index in range(count)]
    def ff(rows, prefix): return linear(swiglu(linear(rows,prefix+".linear_in")),prefix+".linear_out")
    frequencies = [1000*timestep*math.exp(-math.log(10000)*index/4) for index in range(4)]
    embedded = [[math.cos(x) for x in frequencies]+[math.sin(x) for x in frequencies]]
    time = linear([[silu(x) for x in row] for row in linear(embedded,"time_guidance_embed.timestep_embedder.linear_1")],
        "time_guidance_embed.timestep_embedder.linear_2")
    activated = [[silu(x) for x in time[0]]]
    im = chunks(linear(activated,"double_stream_modulation_img.linear")[0],6)
    tm = chunks(linear(activated,"double_stream_modulation_txt.linear")[0],6)
    sm = chunks(linear(activated,"single_stream_modulation.linear")[0],3)
    image = linear(image,"x_embedder")
    text = linear(text,"context_embedder")
    p = "transformer_blocks.0"
    normalized_image = modulate(norm(image),im[0],im[1])
    normalized_text = modulate(norm(text),tm[0],tm[1])
    iq,ik,iv = [linear(normalized_image,p+".attn."+name) for name in ("to_q","to_k","to_v")]
    tq,tk,tv = [linear(normalized_text,p+".attn."+name) for name in ("add_q_proj","add_k_proj","add_v_proj")]
    q = rope(rms(tq,p+".attn.norm_added_q")+rms(iq,p+".attn.norm_q"))
    k = rope(rms(tk,p+".attn.norm_added_k")+rms(ik,p+".attn.norm_k"))
    output = attention(q,k,tv+iv)
    text_count = len(text)
    image = residual(image,linear(output[text_count:],p+".attn.to_out.0"),im[2])
    text = residual(text,linear(output[:text_count],p+".attn.to_add_out"),tm[2])
    image = residual(image,ff(modulate(norm(image),im[3],im[4]),p+".ff"),im[5])
    text = residual(text,ff(modulate(norm(text),tm[3],tm[4]),p+".ff_context"),tm[5])
    value = text+image
    p = "single_transformer_blocks.0.attn"
    packed = linear(modulate(norm(value),sm[0],sm[1]),p+".to_qkv_mlp_proj")
    q,k,v = [[[row[part*16:(part+1)*16][index] for index in range(16)] for row in packed] for part in range(3)]
    attended = attention(rope(rms(q,p+".norm_q")),rope(rms(k,p+".norm_k")),v)
    feedforward = swiglu([row[48:] for row in packed])
    update = linear([a+b for a,b in zip(attended,feedforward)],p+".to_out")
    value = residual(value,update,sm[2])[text_count:]
    scale,shift = chunks(linear(activated,"norm_out.linear")[0],2)
    return linear(modulate(norm(value),shift,scale),"proj_out")
