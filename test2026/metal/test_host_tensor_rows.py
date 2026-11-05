# SPDX-License-Identifier: Apache-2.0
"""XLang native host-row transport checks. Args: LIB SOURCE NEW_OUTPUT."""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
assert not os.path.exists(sys.argv[3]);io=garnet.tensor_io();cases=0;rejected=[]
for shape in [[4,6],[4,2,3],[4,2,2,3]]:
 width=1
 for n in shape[1:]:width*=n
 values=[float(i-23)/7 for i in range(4*width)];source=tensor.tensor(values,shape=shape,dtype=tensor.float32)
 for first in range(4):
  for count in range(1,5-first):
   result=io.rows(source,first,count)
   assert result.tolist()==source.tolist()[first*width:(first+count)*width]
   cases+=1
 assert io.verification_f32_sha256(io.rows(source,0,4))==io.verification_f32_sha256(source)
 for first,count in [(-1,1),(0,0),(0,-1),(4,1),(3,2),(0,9223372036854775807)]:
  failed=False
  try:io.rows(source,first,count)
  except Exception:failed=True
  assert failed;rejected.append([shape,first,count])
special=tensor.tensor([0.0,-0.0,float('inf'),-float('inf'),float('nan'),1e-40],shape=[2,3],dtype=tensor.float32)
assert io.verification_f32_sha256(io.rows(special,0,2))==io.verification_f32_sha256(special);cases+=1
io.close();garnet.shutdown();closed=json.loads(garnet.memory_stats_json())
for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):assert closed[key]==0
with open(sys.argv[3],'x') as f:json.dump(dict(passed=True,cases=cases,rejected=rejected,cleanup=closed,library=garnet.native_library_path(),scope=__doc__),f,indent=2)
print('host_rows_pass',cases,len(rejected),flush=True)
