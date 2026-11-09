# SPDX-License-Identifier: Apache-2.0
"""Native tokenization and host metadata for Klein's single-user prompt profile."""
import json
import _sha2
import math
import os
import garnet
import tensor
from ...generation.runtime import load

def prompt_inputs(checkpoint, prompt, sequence_length=512):
    if not isinstance(prompt,str) or not prompt.strip():
        raise ValueError("a nonempty text prompt is required")
    return prepare_prompt_inputs(checkpoint, sequence_length)(prompt)

def prepare_prompt_inputs(checkpoint, sequence_length=512):
    if sequence_length<1 or sequence_length>512:
        raise ValueError("Klein prompt profile requires 1..512 tokens")
    with open(os.path.join(checkpoint,"tokenizer","chat_template.jinja"),"rb") as source:
        template_digest=_sha2.sha256(source.read()).hexdigest()
    if template_digest!="a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8":
        raise ValueError("this Klein prompt profile has not qualified the checkpoint's chat template")
    tokenizer=garnet.tokenizer()
    tokenizer.open(os.path.join(checkpoint,"tokenizer"))
    pad=tokenizer.token_id("<|endoftext|>")
    if pad<0: raise ValueError("Qwen3 tokenizer lacks required padding")
    with open(os.path.join(checkpoint,"text_encoder","config.json"),"r") as source:
        config=json.load(source)
    width=config["head_dim"]
    theta=config["rope_theta"]
    cosine=[]
    sine=[]
    frequencies=[theta**(-2*index/width) for index in range(width//2)]
    for position in range(sequence_length):
        angles=[position*frequency for frequency in frequencies]
        cosine += [math.cos(angle) for angle in angles]*2
        sine += [math.sin(angle) for angle in angles]*2
    rope_shape=[1,sequence_length,1,width]
    cosine=tensor.tensor(cosine,shape=rope_shape,dtype=tensor.float32)
    sine=tensor.tensor(sine,shape=rope_shape,dtype=tensor.float32)
    masked=[-3e38]*sequence_length
    visible=[0.0]*sequence_length

    def inputs(prompt):
        if not isinstance(prompt,str) or not prompt.strip():
            raise ValueError("a nonempty text prompt is required")
        # Exact checkpoint template for one user message with thinking disabled.
        rendered="<|im_start|>user\n"+prompt+"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids=list(tokenizer.encode(rendered))[:sequence_length]
        count=len(ids)
        if not count: raise ValueError("Qwen3 tokenizer lacks required prompt tokens")
        ids += [pad]*(sequence_length-count)
        mask=[]
        for query in range(sequence_length):
            allowed=min(count,query+1)
            mask.extend(visible[:allowed]);mask.extend(masked[allowed:])
        return [tensor.tensor(ids,shape=[1,sequence_length],dtype=tensor.int32),cosine,sine,
            tensor.tensor(mask,shape=[1,1,sequence_length,sequence_length],dtype=tensor.float32)],count

    return inputs

def load_prompt_encoder(checkpoint, cache, sequence_length=512, backend="tensorrt", model_session=None):
    source=os.path.join(os.path.dirname(__file__),"text_encoder.py")
    loader = model_session.load if model_session is not None else load
    return loader(source, "Flux2TextEncoder", cache,
        [[1,sequence_length],[1,sequence_length,1,128],[1,sequence_length,1,128],[1,1,sequence_length,sequence_length]],
        weights=os.path.join(checkpoint,"text_encoder"), backend=backend,
        dtypes=["int32","float32","float32","float32"])
