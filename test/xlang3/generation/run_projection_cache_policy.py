# SPDX-License-Identifier: Apache-2.0
"""Explicit graph cache hints preserve values, bypass hits, and survive reload."""
import json,os,struct,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
import garnet,tensor
from xModel.generation.runtime import load,execute
root=os.path.abspath(sys.argv[1]);weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
header={'projection.weight':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]}}
raw=json.dumps(header).encode();raw+=b' '*((-len(raw))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(raw))+raw+struct.pack('<4H',0x3f80,0x3f00,0xbf00,0x3f80))
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
source=os.path.join(os.path.dirname(__file__),'fixtures','projection_cache_model.py')
options={'weights':weights,'builder_workspace_mb':64,'linear_kernel_provider':'registered_bf16'}
pool=garnet.cuda_engine_pool();pool.enter()
def stats():
    # Native pool diagnostics omit this section until the first upload.
    return json.loads(pool.stats_json()).get('external_projection_staging',{
        'device_cached_weight_bytes':0,'device_cache_hits':0,'uploaded_weight_bytes':0})
def check(model,scale):
    x=[scale*v for v in [1.,2.,3.,4.]]
    result=execute(model,[tensor.tensor(x,shape=[2,2],dtype=tensor.float32)])
    assert garnet.tensor_to_cpu(result).tolist()==[scale*v for v in [2.,1.5,5.,2.5]]
try:
    for repeat in (False,True):
        cached=load(source,'Cached',os.path.join(root,'cached'),[[2,2]],**options)
        uncached=load(source,'Uncached',os.path.join(root,'uncached'),[[2,2]],**options)
        try:
            if repeat:assert uncached.runtime_status()['forbidden_path_counters']['graph_cache_hits']==1
            before=stats();check(uncached,1);check(uncached,2);after=stats()
            assert after['device_cached_weight_bytes']==0 and after['device_cache_hits']==before['device_cache_hits'],after
            assert after['uploaded_weight_bytes']-before['uploaded_weight_bytes']==16,after
            check(cached,1);before=stats();assert before['device_cached_weight_bytes']==8,before
            check(cached,2);assert stats()['device_cache_hits']==before['device_cache_hits']+1
            before=stats();check(uncached,-1);after=stats()
            assert after['device_cache_hits']==before['device_cache_hits'] and after['uploaded_weight_bytes']-before['uploaded_weight_bytes']==8,after
        finally:cached.release_runtime();uncached.release_runtime()
        assert stats()['device_cached_weight_bytes']==0
    rejected=False
    try:model=load(source,'Invalid',os.path.join(root,'invalid'),[[2,2]],**options)
    except Exception as error:rejected='cache_weights' in str(error)
    else:model.release_runtime()
    assert rejected,'cache_weights must reject non-boolean hints'
finally:pool.leave();pool.close()
after=stats();assert after['registered_checkpoint_bytes']==0 and after['active_device_pools']==0,after
print('Graph projection cache policy passed exact values, bypass/admission, serialized reload, invalid hints and cleanup',flush=True)
