# SPDX-License-Identifier: Apache-2.0
"""Symbolic source/graph checks only; this does not validate checkpoint inference."""
import os
import sys
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..")))
import tensor
from xModel.flux2.klein_4b import transformer as flux

def capture(double_layers, single_layers):
    config = {"guidance_embeds":False,"num_attention_heads":2,"attention_head_dim":8,
        "axes_dims_rope":[2,2,2,2],"num_layers":double_layers,"num_single_layers":single_layers,
        "in_channels":4,"joint_attention_dim":12,"mlp_ratio":3.0,"eps":0.000001,
        "timestep_guidance_channels":8,"patch_size":1}
    inputs = [tensor.input(name,shape=shape,dtype=tensor.float32) for name,shape in [
        ("image",[1,4,4]),("text",[1,3,12]),("time",[1]),
        ("cosine",[1,7,1,8]),("sine",[1,7,1,8])]]
    graph = flux.Flux2Transformer(*inputs,{},config)
    nodes = graph.inspect()
    seen = set()
    for node in nodes:
        # Floating point inputs are literal scalar operands, not node IDs.
        assert all(isinstance(value,float) or value in seen for value in node["inputs"]), node
        seen.add(node["id"])
    operations = [node for node in nodes if node["provider"]]
    names = [node["name"] for node in operations]
    assert names.count("scaled_dot_product_attention") == double_layers+single_layers, names
    assert names.count("rotary_interleaved") == 2*(double_layers+single_layers), names
    assert names.count("swiglu") == 2*double_layers+single_layers, names
    assert names.count("sinusoidal_embedding") == 1, names
    for node in operations:
        assert node["provider"] in ("garnet","cpu"), node
        assert node["regions"][0]["boundary"] == "required", node
    return len(operations)

one = capture(1,1)
two = capture(2,2)
assert two > one, (one,two)
print("FLUX.2 source and graph capture passed",one,two,"operations; checkpoint inference unvalidated",flush=True)
