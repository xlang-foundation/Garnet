# SPDX-License-Identifier: Apache-2.0
"""Opt-in cached preparation: fresh inputs, join-before-release and errors."""
import os,sys,json,struct,time
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
source=os.path.join(os.path.dirname(os.path.abspath(__file__)),'async_weight_fixture.py')
weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
values=[((i*7)%17-8)/8 for i in range(64)];raw=struct.pack('<'+'f'*64,*values)
header=json.dumps({'proj.weight':{'dtype':'F32','shape':[8,8],'data_offsets':[0,len(raw)]}}).encode();header+=b' '*((-len(header))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(header))+header+raw)
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
def load(asynchronous,cache='cache'):
    return garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',weights=weights,
        cache_dir=os.path.join(root,cache),backend='tensorrt',input_shapes=[[1,8]],input_dtypes=['float32'],
        compile={'builder_optimization_level':0,'prepare_async':asynchronous})
def call(model,offset=0):
    return model.forward({'inputs':[tensor.tensor([i+offset for i in range(8)],shape=[1,8],dtype=tensor.float32)]})
def check(model,offset=0):
    result=call(model,offset);assert result['status']=='ok',result
    expected=[sum((i+offset)*values[row*8+i] for i in range(8)) for row in range(8)]
    assert garnet.tensor_to_cpu(result['output']).tolist()==expected
    return result['output'],expected
cold=load(True,'cold-'+str(time.perf_counter()))
assert cold.runtime_status()['engine_preparation_pending'],cold.runtime_status()
check(cold);cold.release_runtime()
eager=load(False);check(eager);eager.release_runtime()
model=load(True);status=model.runtime_status()
assert status['ready'] and status['engine_preparation_pending'] and not status['engines_prepared'],status
held,expected=check(model);check(model,3)
status=model.runtime_status();assert status['engines_prepared'] and not status['engine_preparation_pending'],status
model.release_runtime();assert garnet.tensor_to_cpu(held).tolist()==expected
pending=load(True);pending.release_runtime()
status=pending.runtime_status();assert not status['ready'] and not status.get('engine_preparation_pending',False),status
assert call(pending)['status']=='error'
pool=garnet.cuda_engine_pool();pool.configure(1);pool.enter()
try:
    limited=load(True);assert limited.runtime_status()['engine_preparation_pending']
    result=call(limited);assert result['status']=='error' and result['error_code']=='engine_preparation_failed',result
    limited.release_runtime();assert json.loads(pool.stats_json())['engines']==0
finally:pool.leave();pool.close()
print('Native asynchronous preparation passed: cold compilation, cache reuse, fresh inputs, output lifetime, early release, budget error propagation and scope cleanup',flush=True)
