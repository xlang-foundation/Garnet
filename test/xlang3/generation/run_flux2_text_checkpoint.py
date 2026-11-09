# SPDX-License-Identifier: Apache-2.0
"""Actual native Qwen3 conditioning check. No independent pretrained parity claim."""
import json
import math
import os
import struct
import sys
import time
sys.path.insert(0,os.getcwd())
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","..")))
import garnet
from xModel.flux2.klein_4b.prompt import prompt_inputs,load_prompt_encoder

checkpoint=os.path.abspath(sys.argv[1])
cache=os.path.abspath(sys.argv[2])
length=int(sys.argv[3]) if len(sys.argv)>3 else 512
print("Preparing actual Qwen3-4B prompt encoder",length,"tokens",flush=True)
started=time.perf_counter()
model=load_prompt_encoder(checkpoint,cache,length)
print("Prompt encoder ready",time.perf_counter()-started,"seconds",flush=True)
outputs=[]
for prompt in ("A red apple on a wooden table.","A blue boat on a calm lake."):
    inputs,count=prompt_inputs(checkpoint,prompt,length)
    result=model.forward({"inputs":inputs})
    assert result["status"]=="ok",result
    values=garnet.tensor_to_cpu(result["output"])
    assert list(values.shape)==[1,length,7680],values.shape
    data=values.tolist()
    assert all(math.isfinite(value) for value in data)
    assert max(data)-min(data)>0.001
    outputs.append(data)
    print("Actual prompt encoded",count,"tokens",min(data),max(data),flush=True)
assert max(abs(a-b) for a,b in zip(outputs[0],outputs[1]))>0.001
with open(os.path.join(cache,"prompt-embedding-apple.f32"),"wb") as target:
    target.write(struct.pack("<"+"f"*len(outputs[0]),*outputs[0]))
summary={"shape":[1,length,7680],"native":True,"prompt_difference":max(abs(a-b) for a,b in zip(outputs[0],outputs[1])),
    "scope":"actual-checkpoint finite and prompt-sensitive conditioning; independent pretrained parity unvalidated"}
with open(os.path.join(cache,"checkpoint-sanity.json"),"w") as target: json.dump(summary,target)
assert model.release_runtime()
print("Actual FLUX Qwen3 prompt encoder sanity passed",summary,flush=True)
