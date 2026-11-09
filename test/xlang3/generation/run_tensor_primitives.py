# SPDX-License-Identifier: Apache-2.0
"""Run using XLang3: run_tensor_primitives.py CACHE_ROOT [build|reload]."""
import math
import os
import struct
import sys
sys.path.insert(0, os.getcwd())
import garnet
import tensor

source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensor_primitives.py")
cache = os.path.abspath(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "build"
assert mode in ("build", "reload")

def check(name, shapes, data, output_shape, expected, tolerance=0.00002):
    model = garnet.load_model(source, runtime_mode="compiled_xmodel", entry_function=name,
        input_shapes=shapes, input_dtypes=["float32"] * len(shapes),
        cache_dir=os.path.join(cache, name), backend="tensorrt", precision="bf16",
        compile={"builder_workspace_mb":64, "builder_optimization_level":0})
    status = model.runtime_status()
    assert status["ready"], status
    if mode == "reload":
        assert status["forbidden_path_counters"]["root_x_executions"] == 0, status
    inputs = [tensor.tensor(values, shape=shape, dtype=tensor.float32) for values,shape in zip(data,shapes)]
    result = model.forward({"inputs":inputs})
    assert result["status"] == "ok", result
    output = garnet.tensor_to_cpu(result["output"])
    assert list(output.shape) == output_shape, output.shape
    values = output.tolist()
    assert len(values) == len(expected), values
    for actual,reference in zip(values,expected):
        assert math.isfinite(actual) and abs(actual-reference) <= tolerance, (name,actual,reference)
    assert model.release_runtime()
    print("generation primitive passed", name, mode, flush=True)

rows = [[1.0,2.0,3.0,4.0],[-8.0,2.0,5.0,1.0]]
expected = []
for row in rows:
    mean = sum(row)/len(row)
    variance = sum((value-mean)**2 for value in row)/len(row)
    expected += [(value-mean)/math.sqrt(variance+0.000001) for value in row]
check("norm", [[2,4]], [rows[0]+rows[1]], [2,4], expected)

check("swiglu", [[2,4]], [[-2,3,4,-5,1,-4,7,2]], [2,2],
    [-2/(1+math.exp(2))*4,3/(1+math.exp(-3))*(-5),1/(1+math.exp(-1))*7,-4/(1+math.exp(4))*2])
check("modulate", [[1,2,4],[1,1,4],[1,1,4]],
    [[1,2,3,4,5,6,7,8],[0,1,-1,0.5],[10,-2,3,1]], [1,2,4], [11,2,3,7,15,10,3,13])

# Different angles per channel pair expose split-half/interleaved mistakes.
angles = [0.3,0.3,1.1,1.1,-0.7,-0.7,0.4,0.4]
cosine = [math.cos(value) for value in angles]
sine = [math.sin(value) for value in angles]
values = [1,2,3,4,-5,6,-7,8]
rotated = [-2,1,-4,3,-6,-5,-8,-7]
check("rotary", [[1,2,1,4],[1,2,1,4],[1,2,1,4]], [values,cosine,sine], [1,2,1,4],
    [values[i]*cosine[i]+rotated[i]*sine[i] for i in range(8)])

# A non-square attention problem verifies key/value ordering and softmax axis.
q = [[1,2],[-2,0.5]]
k = [[1,0],[0,1],[-1,-1]]
v = [[3,7],[-2,1],[5,-4]]
expected = []
for query in q:
    scores = [sum(a*b for a,b in zip(query,key))/math.sqrt(2) for key in k]
    weights = [math.exp(score-max(scores)) for score in scores]
    total = sum(weights)
    expected += [sum(weights[index]*v[index][channel] for index in range(3))/total for channel in range(2)]
check("attention", [[1,2,1,2],[1,3,1,2],[1,3,1,2]],
    [[1,2,-2,0.5],[1,0,0,1,-1,-1],[3,7,-2,1,5,-4]], [1,2,1,2], expected)

expected = []
for timestep in [0.0,0.5]:
    frequencies = [1000*timestep*math.exp(-math.log(10000)*index/4) for index in range(4)]
    expected += [math.cos(value) for value in frequencies]+[math.sin(value) for value in frequencies]
check("embedding", [[2]], [[0,0.5]], [2,8], expected, 0.00005)
check("slice_concat", [[2,4],[2,1]], [[1,2,3,4,5,6,7,8],[9,10]], [2,3], [2,3,9,6,7,10])

expected=[]
for row,scale,bias in [([1,2,3,4],2,0.5),([-2,0,2,4],-1,3)]:
    mean=sum(row)/4
    variance=sum((x-mean)**2 for x in row)/4
    expected += [(x-mean)/math.sqrt(variance+0.000001)*scale+bias for x in row]
check("group_norm",[[1,2,2,2],[2],[2]],[[1,2,3,4,-2,0,2,4],[2,-1],[0.5,3]],[1,2,2,2],expected)

data=list(range(1,19))
filters=[1,0,-1,2,0.5,1,2,-1]
expected=[]
for channel,bias in [(0,0.5),(1,-2)]:
    for y in range(2):
        for x in range(2):
            expected.append(sum(data[channel*9+(y+dy)*3+x+dx]*filters[channel*4+dy*2+dx]
                for dy in range(2) for dx in range(2))+bias)
check("conv2d",[[1,2,3,3],[2,1,2,2],[2]],[data,filters,[0.5,-2]],[1,2,2,2],expected)
expected=[]
for y in range(2):
    for x in range(2):
        expected.append(sum(data[t*9+(y+dy)*3+x+dx] for t in range(2) for dy in range(2) for dx in range(2))+0.25)
check("conv3d",[[1,1,2,3,3],[1,1,2,2,2],[1]],[data,[1]*8,[0.25]],[1,1,1,2,2],expected)
check("upsample",[[1,1,2,3]],[[1,2,3,4,5,6]],[1,1,4,6],
    [1,1,2,2,3,3,1,1,2,2,3,3,4,4,5,5,6,6,4,4,5,5,6,6])

angles=[0.3,1.1,0.3,1.1,-0.7,0.4,-0.7,0.4]
cosine=[math.cos(value) for value in angles]
sine=[math.sin(value) for value in angles]
rotated=[-3,-4,1,2,7,-8,-5,6]
check("rotary_split",[[1,2,1,4],[1,2,1,4],[1,2,1,4]],[values,cosine,sine],[1,2,1,4],
    [values[i]*cosine[i]+rotated[i]*sine[i] for i in range(8)])
expected=[3,7,3,7]
for query in ([2,-1],[-1,2]):
    scores=[query[0]/math.sqrt(2),query[1]/math.sqrt(2)]
    probabilities=[math.exp(score-max(scores)) for score in scores]
    denominator=sum(probabilities)
    expected += [(probabilities[0]*a+probabilities[1]*b)/denominator for a,b in zip([3,7],[-2,1])]
check("masked_gqa",[[1,2,2,2],[1,3,1,2],[1,3,1,2],[1,1,2,3]],
    [[1,0,0,1,2,-1,-1,2],[1,0,0,1,-1,-1],[3,7,-2,1,5,-4],
    [0,-1e30,-1e30,0,0,-1e30]],[1,2,2,2],expected)
check("round_bfloat16",[[6]],[[1.001,1.006,-1.006,0,0.3333,100.1]],[6],
    [1.0,1.0078125,-1.0078125,0,0.333984375,100.0],0.000001)
check("scalar_math",[[2,2]],[[-4,0,4,8]],[2,2],[-2,0,2,4])
check("rank_broadcast",[[2,2,3],[2,3]],[[1,2,3,4,5,6,7,8,9,10,11,12],[10,20,30,40,50,60]],
    [2,2,3],[11,22,33,44,55,66,17,28,39,50,61,72])
check("conv2d_asymmetric",[[1,1,4,4],[1,1,3,3]],
    [list(range(1,17)),[1]*9],[1,1,2,2],[54,45,72,54])
check("conv3d_causal",[[1,1,4,1,1],[1,1,3,1,1]],
    [[1,2,3,4],[1,2,3]],[1,1,4,1,1],[3,8,14,20])
def bf16(value):
    bits=struct.unpack("<I",struct.pack("<f",value))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack("<f",struct.pack("<I",bits))[0]
queries=[[0.25,-0.5],[0.75,0.25]]
keys=[[1,0.5],[-0.25,0.75]]
bias=[[0.2,-0.4],[0.3,0.1]]
expected=[]
for query,row_bias in zip(queries,bias):
    scores=[bf16(bf16(sum(a*b for a,b in zip(query,key)))+add) for key,add in zip(keys,row_bias)]
    probabilities=[math.exp(score-max(scores)) for score in scores]
    denominator=sum(probabilities)
    p0,p1=[bf16(value/denominator) for value in probabilities]
    expected += [p0*0.25-p1,p0*2+p1*0.5]
check("unscaled_bf16_attention",[[1,2,1,2],[1,2,1,2],[1,2,1,2],[1,1,2,2]],
    [[0.25,-0.5,0.75,0.25],[1,0.5,-0.25,0.75],[0.25,2,-1,0.5],[0.2,-0.4,0.3,0.1]],
    [1,2,1,2],expected)
check("l2_normalize",[[3,2]],[[3,4,0,0,3e-14,4e-14]],[3,2],[0.6,0.8,0,0,0.03,0.04])
check("reduce_mean",[[2,3,2]],[[1,2,3,4,5,6,-1,1,-2,2,-3,3]],[2,1,2],[3,4,-2,2])
check("l2_channel_axis",[[1,2,1,2]],[[3,0,4,0]],[1,2,1,2],[0.6,0,0.8,0])

for name,shape in [("invalid_slice",[2,4]),("invalid_axis",[2,4]),("swiglu",[2,3])]:
    failed = False
    try:
        model = garnet.load_model(source,runtime_mode="compiled_xmodel",entry_function=name,
            input_shapes=[shape],input_dtypes=["float32"],backend="tensorrt",precision="bf16",
            cache_dir=os.path.join(cache,"reject_"+name))
        failed = not model.runtime_status()["ready"]
        model.release_runtime()
    except Exception:
        failed = True
    assert failed, "invalid primitive contract accepted: "+name
print("generation tensor primitive numerical suite passed", mode, flush=True)
