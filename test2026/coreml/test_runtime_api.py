# SPDX-License-Identifier: Apache-2.0
"""XLang integration only; native C++/Objective-C++ executes all Core ML calls.

Args: LIBRARY_ROOT SOURCE_ROOT COMPILED_FIXTURE OUTPUT_JSON.
No trained-model correctness, serving, actual ANE placement or speed claim.
"""
import sys,json,os
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
assert 'coreml_stateful_optional' in garnet.backend_info()
assert garnet.native_library_path()==sys.argv[1]+'/libgarnet.dylib'
assert not os.path.exists(sys.argv[4])
shape=[1,32,1,16]
profile=dict(compute_units='cpu_ane',maximum_io_bytes=2048,maximum_state_bytes=1024,
             require_ane_plan=False,inputs=dict(x=dict(dtype='float16',maximum_shape=shape)),
             outputs=dict(y=dict(dtype='float16',maximum_shape=shape)))
checks=0
def check_output(session,value,expected):
 global checks
 x=tensor.tensor([value]*512,shape=shape,dtype=tensor.float32)
 out=session.predict_float32(dict(x=x))['y'].tolist()
 assert len(out)==512 and all(item==expected for item in out)
 checks+=512
def reject(call):
 global checks
 rejected=False
 try:call()
 except Exception:rejected=True
 assert rejected
 checks+=1
session=garnet.coreml_session()
assert session.open(sys.argv[3],json.dumps(profile))
check_output(session,2,1)
saved=session.snapshot_state()
check_output(session,3,2.5)
assert session.restore_state(saved)
check_output(session,2,2)
assert session.reset()
check_output(session,2,1)
reject(lambda:session.predict(dict(x=tensor.tensor([1.0]*512,shape=shape,dtype=tensor.float32))))
reject(lambda:session.predict(dict()))
reject(lambda:session.predict_float32(dict(x=tensor.tensor([1]*512,shape=shape,dtype=tensor.int32))))
reject(lambda:session.predict_float32(dict(x=tensor.tensor([1.0]*544,shape=[1,32,1,17],dtype=tensor.float32))))
reject(lambda:session.restore_state(dict()))
reject(lambda:session.restore_state(dict(accumulator=tensor.tensor([1.0]*512,shape=shape,dtype=tensor.float32))))
check_output(session,2,2)
assert session.reset()
# Independent known half rounding: fp16(0.1) /2 is exactly this binary value.
check_output(session,.1,.04998779296875)
report=json.loads(session.report_json())
assert report['predictions']==6 and report['declared_state_bytes']==1024
assert session.close() and session.close()
reject(lambda:session.predict(dict()))
with open(sys.argv[4],'x') as stream:
 json.dump(dict(passed=True,checks=checks,backend=garnet.backend_info(),
   library=garnet.native_library_path(),report=report,scope=__doc__),stream,indent=2)
print('coreml_runtime_api',checks,'passed')
