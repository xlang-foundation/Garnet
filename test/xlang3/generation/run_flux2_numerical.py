# SPDX-License-Identifier: Apache-2.0
"""XLang3-only tiny FLUX.2 transformer versus independent scalar arithmetic."""
import math
import os
import sys
sys.path.insert(0, os.getcwd())
import garnet
import tensor
from flux2_reference import fixture,forward

test_root = os.path.abspath(sys.argv[1])
mode = sys.argv[2] if len(sys.argv)>2 else "build"
assert mode in ("build","reload")
checkpoint = os.path.join(test_root,"fixture")
weights,shapes = fixture(checkpoint)
image = [[math.sin(index*0.31) for index in range(4)], [math.cos(index*0.47) for index in range(4)]]
text = [[0.2*math.sin(index*0.13+row) for index in range(12)] for row in range(3)]
angles = [[0.11*position+0.23*(channel//2) for channel in range(8)] for position in range(5)]
cosine = [[math.cos(value) for value in row] for row in angles]
sine = [[math.sin(value) for value in row] for row in angles]
timestep = 0.375
expected = [value for row in forward(weights,shapes,image,text,timestep,cosine,sine) for value in row]
repository = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",".."))
source = os.path.join(repository,"xModel","flux2","klein_4b","transformer.py")
model = garnet.load_model(source,runtime_mode="compiled_xmodel",entry_function="Flux2Transformer",
    input_shapes=[[1,2,4],[1,3,12],[1],[1,5,1,8],[1,5,1,8]],input_dtypes=["float32"]*5,
    weights=checkpoint,cache_dir=os.path.join(test_root,"engine"),backend="tensorrt",precision="bf16",
    compile={"builder_workspace_mb":64,"builder_optimization_level":0})
status = model.runtime_status()
assert status["ready"],(status["error_code"],status["error_message"])
if mode=="reload": assert status["forbidden_path_counters"]["root_x_executions"]==0,status
def flatten(rows): return [value for row in rows for value in row]
inputs = [tensor.tensor(data,shape=shape,dtype=tensor.float32) for data,shape in [
    (flatten(image),[1,2,4]),(flatten(text),[1,3,12]),([timestep],[1]),
    (flatten(cosine),[1,5,1,8]),(flatten(sine),[1,5,1,8])]]
result = model.forward({"inputs":inputs})
assert result["status"]=="ok",result
output = garnet.tensor_to_cpu(result["output"])
assert list(output.shape)==[1,2,4],output.shape
actual = output.tolist()
assert len(actual)==len(expected)
max_error = max(abs(a-b) for a,b in zip(actual,expected))
for a,b in zip(actual,expected): assert math.isfinite(a) and abs(a-b)<0.0002,(actual,expected,max_error)
assert model.release_runtime()
print("FLUX.2 tiny transformer scalar parity passed",mode,"max_abs_error",max_error,flush=True)

# Editing appends immutable reference tokens and returns only the generated
# image's velocity. The scalar oracle still attends over the complete stream.
model=garnet.load_model(os.path.join(repository,"xModel","flux2","klein_4b","editing.py"),
    runtime_mode="compiled_xmodel",entry_function="Flux2EditingTransformer",
    input_shapes=[[1,1,4],[1,1,4],[1,3,12],[1],[1,5,1,8],[1,5,1,8]],input_dtypes=["float32"]*6,
    weights=checkpoint,cache_dir=os.path.join(test_root,"editing-engine"),backend="tensorrt",precision="bf16",
    compile={"builder_workspace_mb":64,"builder_optimization_level":0})
status=model.runtime_status()
assert status["ready"],(status["error_code"],status["error_message"])
if mode=="reload": assert status["forbidden_path_counters"]["root_x_executions"]==0,status
editing_inputs=[tensor.tensor(row,shape=[1,1,4],dtype=tensor.float32) for row in image]+inputs[1:]
result=model.forward({"inputs":editing_inputs})
assert result["status"]=="ok",result
output=garnet.tensor_to_cpu(result["output"])
assert list(output.shape)==[1,1,4]
assert all(math.isfinite(a) and abs(a-b)<0.0002 for a,b in zip(output.tolist(),expected[:4]))
assert model.release_runtime()
print("FLUX.2 editing reference-token attention and output slicing parity passed",mode,flush=True)
