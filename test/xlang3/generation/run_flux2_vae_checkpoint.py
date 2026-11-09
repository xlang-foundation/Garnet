# SPDX-License-Identifier: Apache-2.0
"""Actual-checkpoint VAE smoke check, using XLang3 and native Garnet only.

This checks real checkpoint loading and numerical sanity. It is not independent
pretrained parity or a prompt-to-image quality test.
"""
import json
import math
import os
import sys
sys.path.insert(0,os.getcwd())
import garnet
import tensor

checkpoint = os.path.abspath(sys.argv[1])
cache = os.path.abspath(sys.argv[2])
repository = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
source = os.path.join(repository,"xModel","flux2","klein_4b","vae.py")
model = garnet.load_model(source,runtime_mode="compiled_xmodel",entry_function="Flux2VAEDecode",
    weights=os.path.join(checkpoint,"diffusion_pytorch_model.safetensors"),
    input_shapes=[[1,32,2,2]],input_dtypes=["float32"],backend="tensorrt",precision="bf16",
    cache_dir=cache,compile={"builder_workspace_mb":64,"builder_optimization_level":0})
status = model.runtime_status()
assert status["ready"], (status["error_code"],status["error_message"])
values = [0.1*math.sin(index*0.31) for index in range(128)]
result = model.forward({"inputs":[tensor.tensor(values,shape=[1,32,2,2],dtype=tensor.float32)]})
assert result["status"]=="ok",result
output = garnet.tensor_to_cpu(result["output"])
assert list(output.shape)==[1,3,16,16],output.shape
decoded = output.tolist()
assert all(math.isfinite(value) for value in decoded)
assert max(decoded)-min(decoded)>0.01,(min(decoded),max(decoded))
assert max(abs(value) for value in decoded)<100,(min(decoded),max(decoded))
summary={"checkpoint":checkpoint,"output_shape":list(output.shape),"min":min(decoded),"max":max(decoded),
    "mean":sum(decoded)/len(decoded),"scope":"actual-checkpoint VAE sanity; pretrained parity and prompt-to-image unvalidated"}
with open(os.path.join(cache,"checkpoint-sanity.json"),"w") as destination: json.dump(summary,destination)
assert model.release_runtime()
print("FLUX.2 actual-checkpoint VAE decode sanity passed",summary,flush=True)
