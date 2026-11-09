# SPDX-License-Identifier: Apache-2.0
"""Klein's Qwen3-4B conditioning: concatenate hidden states 9, 18, and 27."""
# garnet-dependency: ../../qwen3/hidden_states.py
# garnet-dependency: ../../generation/tensor_ops.py
from ...generation import tensor_ops as O
from ...qwen3 import hidden_states as qwen
T=O.T
GARNET_MODEL_SPEC={"arguments":[
    {"name":"input_ids","kind":"tensor"},
    {"name":"cosine","kind":"tensor"},
    {"name":"sine","kind":"tensor"},
    {"name":"attention_mask","kind":"tensor"},
    {"name":"weights","kind":"weights"},
    {"name":"config","kind":"config"},
]}

@T.fusion(name="flux2_qwen3_prompt_encoder",boundary="required")
def Flux2TextEncoder(input_ids, cosine, sine, attention_mask, weights, config):
    if config["hidden_size"] != 2560 or config["num_hidden_layers"] != 36:
        raise ValueError("FLUX.2 Klein 4B conditioning requires its Qwen3-4B checkpoint")
    return qwen.hidden_states(input_ids,cosine,sine,attention_mask,config,(9,18,27))
