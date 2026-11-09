# SPDX-License-Identifier: Apache-2.0
"""Real streamed weights: LRU eviction, active leases, rejection and cleanup."""
import os,sys,json,struct
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
source=os.path.join(root,'weighted.py');weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
with open(source,'w') as f:f.write("import garnet\nT=garnet.tensor()\nGARNET_MODEL_SPEC={'arguments':[{'name':'x','kind':'tensor'},{'name':'weights','kind':'weights'}]}\n@T.fusion(name='pool_weight_fixture')\ndef forward(x,weights):return x*T.unary_op('linear',weight_name='proj.weight')\n")
values=[((i*7)%17-8)/8 for i in range(64)];raw=struct.pack('<'+'f'*64,*values)
header=json.dumps({'proj.weight':{'dtype':'F32','shape':[8,8],'data_offsets':[0,len(raw)]}}).encode();header+=b' '*((-len(header))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(header))+header+raw)
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
def load(name):
    return garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',weights=weights,
        cache_dir=os.path.join(root,name),backend='tensorrt',input_shapes=[[1,8]],input_dtypes=['float32'],
        compile={'builder_optimization_level':0})
def check(model):
    assert model.runtime_status()['ready'],model.runtime_status()
    result=model.forward({'inputs':[tensor.tensor(list(range(8)),shape=[1,8],dtype=tensor.float32)]})
    assert result['status']=='ok',result
    expected=[sum(i*values[row*8+i] for i in range(8)) for row in range(8)]
    assert garnet.tensor_to_cpu(result['output']).tolist()==expected
def rejected(name):
    try:
        model=load(name);ready=model.runtime_status()['ready'];model.release_runtime()
        assert not ready,'Over-budget model unexpectedly loaded'
    except AssertionError:raise
    except Exception:pass
prime=garnet.cuda_engine_pool();prime.configure(1<<20);prime.enter();model=load('prime');check(model);model.release_runtime()
size=json.loads(prime.stats_json())['streamable_weight_bytes'];assert size>0,'Run with GARNET_TRT_WEIGHT_STREAMING_BUDGET_MIB=1'
prime.leave();prime.close()
pool=garnet.cuda_engine_pool();pool.configure(size);pool.enter()
for name in ('a','b','a'):
    model=load(name);check(model);model.release_runtime()
    stats=json.loads(pool.stats_json());assert stats['engines']==1 and stats['streamable_weight_bytes']==size,stats
assert stats['evictions']==2 and stats['engine_misses']==3,stats
active=load('a');check(active);rejected('b');check(active)
active.release_runtime();pool.leave();pool.close()
small=garnet.cuda_engine_pool();small.configure(size-1);small.enter();rejected('a');small.leave()
assert json.loads(small.stats_json())['engines']==0;small.close()
scope=garnet.cuda_engine_pool()
failed=False
try:garnet.generate('flux2-klein-4b','x',checkpoint=root,cache=root,output=os.path.join(root,'unused.png'),height=1,engine_pool=scope)
except Exception as error:
    assert 'FLUX image dimensions' in str(error),error
    failed=True
assert failed and json.loads(scope.stats_json())['active_scopes']==0
scope.close()
print('Native weighted engine pool passed: nonzero weights, bounded LRU, active-lease protection, budget rejection and generation-error scope cleanup',size,flush=True)
