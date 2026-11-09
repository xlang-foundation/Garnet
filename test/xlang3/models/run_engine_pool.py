# SPDX-License-Identifier: Apache-2.0
"""Native engine residency, released leases, fresh inputs and source invalidation."""
import os,sys,json
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
source=os.path.join(root,'model.py');cache=os.path.join(root,'cache')
def write(operation):
    previous=os.path.getmtime(source) if os.path.isfile(source) else 0
    with open(source,'w') as f:f.write("import garnet\nT=garnet.tensor()\n@T.fusion(name='pool_fixture')\ndef forward(a,b):return a "+operation+" b\n")
    stamp=max(os.path.getmtime(source),previous+2);os.utime(source,(stamp,stamp))
def load():
    model=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',cache_dir=cache,
        backend='tensorrt',precision='bf16',input_shapes=[[2],[2]],input_dtypes=['float32','float32'],
        compile={'builder_workspace_mb':64,'builder_optimization_level':0})
    assert model.runtime_status()['ready'],model.runtime_status()
    return model
def check(model,a,b,expected):
    result=model.forward({'inputs':[tensor.tensor(a,dtype=tensor.float32),tensor.tensor(b,dtype=tensor.float32)]})
    assert result['status']=='ok',result
    out=result['output'];assert garnet.tensor_to_cpu(out).tolist()==expected
    return out
def reject(call):
    failed=False
    try:call()
    except Exception:failed=True
    assert failed
write('+');pool=garnet.cuda_engine_pool();pool.configure(1<<30);pool.enter()
reject(pool.close)
other=garnet.cuda_engine_pool();reject(other.enter);other.close()
first=load();held=check(first,[8,6],[2,3],[10,9]);first.release_runtime()
stats=json.loads(pool.stats_json());assert stats['engines']==1 and stats['contexts']==0,stats
misses=stats['engine_misses'];second=load();check(second,[1,4],[5,7],[6,11]);second.release_runtime()
assert garnet.tensor_to_cpu(held).tolist()==[10,9]
stats=json.loads(pool.stats_json());assert stats['engine_misses']==misses and stats['engine_hits']>0 and stats['contexts']==0,stats
write('-');changed=load();check(changed,[8,6],[2,3],[6,3]);changed.release_runtime()
stats=json.loads(pool.stats_json());assert stats['engines']==1 and stats['engine_misses']==misses+1 and stats['contexts']==0,stats
pool.leave();reject(pool.leave);pool.close();pool.close();reject(pool.enter)
assert json.loads(pool.stats_json())['engines']==0
fresh=load();check(fresh,[4,8],[1,3],[3,5]);fresh.release_runtime()
print('Native engine pool passed: residency, fresh inputs, context retirement, invalidation, released output ownership and lifecycle',flush=True)
