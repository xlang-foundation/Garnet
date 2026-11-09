# SPDX-License-Identifier: Apache-2.0
"""Native converter: RNE ties, module boundaries, retained bytes, no overwrite."""
import json
import os
import struct
import subprocess
import sys
tool=os.path.abspath(sys.argv[1]);root=os.path.abspath(sys.argv[2])
source=os.path.join(root,"source");destination=os.path.join(root,"output")
os.makedirs(source,exist_ok=True)
policy=os.path.join(root,"policy.json")
with open(policy,"w") as target: json.dump({"target_dtype":"BF16","keep_fp32_modules":["norm2"]},target)
payload=b"";header={}
entries=[("linear.weight","F32",[7],struct.pack("<7f",1.001,1.006,-1.006,0.3333,1.00390625,1.01171875,-0.0)),
    ("blocks.0.norm2.weight","F32",[2],struct.pack("<2f",0.5,2.75)),
    ("blocks.0.abnorm2.weight","F32",[1],struct.pack("<f",1.00390625)),
    ("embedding.weight","BF16",[2],struct.pack("<2H",0x3f80,0x4000)),
    ("abnorm2.norm2.weight","F32",[1],struct.pack("<f",1.00390625))]
for name,dtype,shape,data in entries:
    header[name]={"dtype":dtype,"shape":shape,"data_offsets":[len(payload),len(payload)+len(data)]}
    payload+=data
text=json.dumps(header,separators=(",",":")).encode("utf8");text+=b" "*((-len(text))%8)
original=struct.pack("<Q",len(text))+text+payload
with open(os.path.join(source,"model.safetensors"),"wb") as target: target.write(original)
with open(os.path.join(source,"config.json"),"w") as target: json.dump({"model_type":"precision-test"},target)
planned=subprocess.run([tool,source,destination,policy,"--plan"],capture_output=True,text=True)
assert planned.returncode==0,planned.stderr
plan=json.loads(planned.stdout)
assert plan["converted_tensors"]==2 and plan["retained_tensors"]==3 and plan["tensor_bytes"]==32,plan
# Use a new empty destination for each invocation, preserving prior evidence.
serial=0
while os.path.exists(destination):
    serial+=1;destination=os.path.join(root,"output-"+str(serial))
result=subprocess.run([tool,source,destination,policy],capture_output=True,text=True)
assert result.returncode==0,(result.stdout,result.stderr)
with open(os.path.join(destination,"model.safetensors"),"rb") as source_file: prepared=source_file.read()
length=struct.unpack("<Q",prepared[:8])[0]
metadata=json.loads(prepared[8:8+length].decode("utf8"))
def data(name):
    start,end=metadata[name]["data_offsets"]
    return prepared[8+length+start:8+length+end]
assert metadata["linear.weight"]["dtype"]=="BF16"
assert list(struct.unpack("<7H",data("linear.weight")))==[0x3f80,0x3f81,0xbf81,0x3eab,0x3f80,0x3f82,0x8000]
assert metadata["blocks.0.norm2.weight"]["dtype"]=="F32" and data("blocks.0.norm2.weight")==entries[1][3]
assert metadata["blocks.0.abnorm2.weight"]["dtype"]=="BF16"
assert data("embedding.weight")==entries[3][3]
assert metadata["abnorm2.norm2.weight"]["dtype"]=="F32" and data("abnorm2.norm2.weight")==entries[4][3]
again=subprocess.run([tool,source,destination,policy],capture_output=True,text=True)
assert again.returncode!=0,"nonempty checkpoint destination was overwritten"
with open(os.path.join(destination,"model.safetensors"),"rb") as source_file: assert source_file.read()==prepared
with open(os.path.join(source,"model.safetensors"),"rb") as source_file: assert source_file.read()==original
mapped=destination+'-mapped'
config_file=os.path.join(source,'config.json')
mapping={'linear.weight':'renamed.projection.weight','embedding.weight':'renamed.embedding.weight'}
with open(policy,'w') as target:json.dump({'target_dtype':'BF16','keep_fp32_modules':[],'config_file':config_file,'tensor_map':mapping},target)
result=subprocess.run([tool,os.path.join(source,'model.safetensors'),mapped,policy],capture_output=True,text=True)
assert result.returncode==0,(result.stdout,result.stderr)
with open(os.path.join(mapped,'diffusion_pytorch_model.safetensors.index.json')) as f:index=json.load(f)
assert set(index['weight_map'])==set(mapping.values()),index
with open(os.path.join(mapped,index['weight_map']['renamed.embedding.weight']),'rb') as f:raw=f.read()
length=struct.unpack('<Q',raw[:8])[0];meta=json.loads(raw[8:8+length].decode('utf8'))
assert set(meta)-{'__metadata__'}==set(mapping.values()),meta
start,end=meta['renamed.embedding.weight']['data_offsets'];assert raw[8+length+start:8+length+end]==entries[3][3]
with open(os.path.join(mapped,'config.json')) as f:assert json.load(f)['model_type']=='precision-test'
for invalid in ({'missing.weight':'renamed.weight'},{'linear.weight':'same','embedding.weight':'same'},{'linear.weight':''}):
    with open(policy,'w') as target:json.dump({'target_dtype':'BF16','keep_fp32_modules':[],'config_file':config_file,'tensor_map':invalid},target)
    result=subprocess.run([tool,source,mapped+'-invalid',policy,'--plan'],capture_output=True,text=True)
    assert result.returncode!=0,invalid
print("Native checkpoint precision preparation passed: RNE ties, protected module boundaries, retained bytes, source preservation and destination protection",flush=True)
