# SPDX-License-Identifier: Apache-2.0
"""Reuse a compiled model after fenced context retirement; preserve held outputs."""
import json,os,struct,sys,threading
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
import garnet,tensor
from xModel.generation.runtime import load,execute
from xModel.generation.session import GenerationSession
root=os.path.abspath(sys.argv[1]);weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
header={'projection.weight':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]},'projection.bias':{'dtype':'BF16','shape':[2],'data_offsets':[8,12]}}
raw=json.dumps(header).encode();raw+=b' '*((-len(raw))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(raw))+raw+struct.pack('<6H',0x3f80,0x3b80,0xbf80,0xbb80,0xbf80,0x3f80))
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
source=os.path.join(os.path.dirname(__file__),'fixtures','linear_accumulation_model.py');cache=os.path.join(root,'engine')
options={'weights':weights,'builder_workspace_mb':64,'linear_kernel_provider':'registered_bf16'}
def rejected(fn):
    failed=False
    try:fn()
    except Exception:failed=True
    assert failed
raw_model=load(source,'FloatOutput',cache,[[4,2]],**options)
try:rejected(lambda:raw_model.trim_execution_contexts())
finally:raw_model.release_runtime()
session=GenerationSession(maximum_models=1);session.enter();held=None
try:
    errors=[]
    def other_thread():
        try:session.check_active()
        except Exception:errors.append(True)
    thread=threading.Thread(target=other_thread);thread.start();thread.join();assert errors==[True]
    for sign in (1.0,-1.0):
        model=session.load(source,'FloatOutput',cache,[[4,2]],**options)
        rejected(lambda:session.close())
        rejected(lambda:session.load(source,'FloatOutput',cache,[[4,2]],**options))
        values=[sign*x for x in [1.,1.,1.,-1.,-1.,1.,-1.,-1.]]
        expected=[v for x,y in zip(values[::2],values[1::2]) for v in (x+y/256-1,-x-y/256+1)]
        result=execute(model,[tensor.tensor(values,shape=[4,2],dtype=tensor.float32)])
        assert garnet.tensor_to_cpu(result).tolist()==expected
        model.release_runtime();assert model.runtime_status()['ready']
        rejected(lambda:execute(model,[]))
        stats=session.stats();assert stats['models']==1 and stats['active_leases']==0 and stats['engine_pool']['contexts']==0,stats
        if held is not None:assert garnet.tensor_to_cpu(held).tolist()==saved
        held=result;saved=expected
    rejected(lambda:session.load(source,'FloatOutput',cache+'-other',[[4,2]],**options))
finally:session.close()
session.close();stats=session.stats();assert stats['closed'] and stats['models']==0 and stats['engine_pool']['engines']==0
external=stats['engine_pool']['external_projection_staging'];assert external['mapped_files']==0 and external['registered_checkpoint_bytes']==0 and external['active_device_pools']==0,external
assert garnet.tensor_to_cpu(held).tolist()==saved
resident=GenerationSession(maximum_models=1,keep_execution_contexts=True);resident.enter()
held_resident=None
try:
    for index in range(2):
        model=resident.load(source,'FloatOutput',cache,[[4,2]],**options)
        rejected(lambda:resident.clear_device_cache())
        previous_hits=resident.stats()['engine_pool']['external_projection_staging']['device_cache_hits']
        current=[v*(index+1) for v in values]
        reference=[v for x,y in zip(current[::2],current[1::2]) for v in (x+y/256-1,-x-y/256+1)]
        result=execute(model,[tensor.tensor(current,shape=[4,2],dtype=tensor.float32)])
        assert garnet.tensor_to_cpu(result).tolist()==reference
        model.release_runtime(keep_device_cache=index==0)
        current_stats=resident.stats()['engine_pool'];assert current_stats['contexts']==1
        assert current_stats['external_projection_staging']['device_cached_weight_bytes']==(8 if index==0 else 0)
        if index==1:assert current_stats['external_projection_staging']['device_cache_hits']==previous_hits+1
        if held_resident is not None:assert garnet.tensor_to_cpu(held_resident).tolist()==resident_reference
        held_resident=result;resident_reference=reference
    resident.clear_device_cache()
    component=resident.load_component(source,'FloatOutput',cache,[[4,2]],**options)
    assert resident.stats()['models']==1
    result=execute(component,[tensor.tensor(current,shape=[4,2],dtype=tensor.float32)])
    assert garnet.tensor_to_cpu(result).tolist()==resident_reference
    component.release_runtime();assert resident.stats()['engine_pool']['contexts']==0
    assert garnet.tensor_to_cpu(held_resident).tolist()==resident_reference
finally:resident.close()
assert resident.stats()['engine_pool']['contexts']==0
assert garnet.tensor_to_cpu(held_resident).tolist()==resident_reference

switching=GenerationSession(maximum_models=2);switching.enter()
try:
    alternate=cache+'-context-slot'
    for selected in (cache,alternate,cache):
        active=switching.load(source,'FloatOutput',selected,[[4,2]],keep_execution_contexts=True,context_slot='base',**options)
        other=cache if selected==alternate else alternate
        rejected(lambda:switching.load(source,'FloatOutput',other,[[4,2]],keep_execution_contexts=True,context_slot='base',**options))
        result=execute(active,[tensor.tensor(current,shape=[4,2],dtype=tensor.float32)])
        assert garnet.tensor_to_cpu(result).tolist()==resident_reference
        active.release_runtime()
        assert switching.stats()['engine_pool']['contexts']==1
finally:switching.close()

print('Native generation session passed profile bounds, thread/lease guards, exact reload outputs and complete residency cleanup',flush=True)
