"""Normal decoder admission stays batch1/2; batch4 needs its native extension.

Arguments: native build without chunk_kv_inplace, source root, real config JSON.
No model weights are opened by this rejection gate.
"""
import sys,json
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.cpu_decoder import CPUDecoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
from xModel.qwen38.vl_27b.metal_batch_study import create_decoder
assert 'chunk_kv_inplace' not in garnet.backend_info()
for cls in (CPUDecoder,MetalDecoder):
    rejected=False
    try:cls('/nonexistent-admission-check',{},4,512)
    except ValueError:rejected=True
    assert rejected
with open(sys.argv[3]) as stream:config=json.load(stream)
rejected=False
try:create_decoder('/nonexistent-admission-check',config,512,4,8)
except ValueError as error:rejected='in-place chunk-cache' in str(error)
assert rejected
print('batch_study_admission PASS normal batch4 rejected and study extension required before weight open',flush=True)
garnet.shutdown()
