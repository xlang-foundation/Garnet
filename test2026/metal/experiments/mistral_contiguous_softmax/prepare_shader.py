"""Compose the pinned original shader and a separate experimental kernel. No build."""
from pathlib import Path
import hashlib
import json
import sys

source=Path(sys.argv[1]).resolve()
out=Path(sys.argv[2]).resolve()
candidate=Path(__file__).resolve().parent/'candidate-softmax2.metal'
header=source/'src/backends/metal/chunk_attention_mps_shader.h'
assert hashlib.sha256(header.read_bytes()).hexdigest()=='350766841d590b0c57b71c83a9f08357d30da34320446c6cfead5321b4c86088'
assert not out.exists(), 'Prepare a fresh study; do not overwrite evidence.'
out.mkdir(parents=True)
shader=header.read_text().split('R"METAL(',1)[1].split(')METAL"',1)[0]
pre='''#include <metal_stdlib>
using namespace metal;
#define GARNET_CHUNK_ATTENTION_INDEX32
#define GARNET_CHUNK_ATTENTION_BUCKETS
struct ChunkAttentionParams {ulong batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
struct BoundedChunkAttentionParams {uint batch,chunk,heads,kvHeads,dim,capacity,qStride,kStride,vStride,kcStride,vcStride,sStride,aStride,gStride,oStride,gated;};
'''
context=(source/'src/backends/metal/metal_context.mm').read_text()
for line in pre.splitlines():
    if line.startswith('struct '): assert line in context
combined=pre+shader+candidate.read_text()
assert hashlib.sha256(combined.encode()).hexdigest()=='b3fb8080347141b8384a0dfe128657095353fc2f0fb30b9a1c15defe62da249a'
(out/'combined2.metal').write_text(combined)
(out/'shader-bindings1.json').write_text(json.dumps(dict(
    original_header=str(header),original_header_sha256=hashlib.sha256(header.read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    combined_sha256=hashlib.sha256(combined.encode()).hexdigest(),
    native_integration=False,serving_speed_claim=False),indent=2)+'\n')
print(out/'combined2.metal')
