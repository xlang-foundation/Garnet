# SPDX-License-Identifier: Apache-2.0
"""Bias-cancellation case distinguishes pre-bias BF16 rounding from FP32 output."""
import json,os,struct,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..')))
import garnet,tensor
from xModel.generation.runtime import load,execute
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
header={'projection.weight':{'dtype':'BF16','shape':[2,2],'data_offsets':[0,8]},'projection.bias':{'dtype':'BF16','shape':[2],'data_offsets':[8,12]}}
raw=json.dumps(header).encode();raw+=b' '*((-len(raw))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(raw))+raw+struct.pack('<6H',0x3f80,0x3b80,0xbf80,0xbb80,0xbf80,0x3f80))
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
source=os.path.join(os.path.dirname(__file__),'fixtures','linear_accumulation_model.py')
values=[1.,1.,1.,-1.,-1.,1.,-1.,-1.];input=tensor.tensor(values,shape=[4,2],dtype=tensor.float32)
exact=[value for x,y in zip(values[::2],values[1::2]) for value in (x+y/256-1,-x-y/256+1)]
rounded=[1/256,-1/256,-1/256,1/256,-2.,2.,-2.,2.]
if not os.environ.get('GARNET_CUDA_REGISTERED_WEIGHT_BUDGET_MIB'):raise ValueError('Set the explicit host registration budget before starting XLang3 (CTest uses 4 MiB)')
for provider in (None,'streamed_bf16','registered_bf16'):
    for entry,expected in [('Preserved',rounded),('FloatOutput',exact),('Legacy',None)]:
        model=load(source,entry,os.path.join(root,(provider or 'native')+'-'+entry),[[4,2]],weights=weights,builder_workspace_mb=64,linear_kernel_provider=provider)
        try:actual=garnet.tensor_to_cpu(execute(model,[input])).tolist()
        finally:model.release_runtime()
        if entry=='Legacy':
            # The default path retains its previous rounding. In the first
            # row, ties round the dot product to 1/-1 before bias addition.
            assert actual[:2]==[0.,0.],(provider,entry,actual)
        else:assert actual==expected,(provider,entry,actual,expected)
        if entry=='FloatOutput' and provider is not None:
            cache=os.path.join(root,provider+'-'+entry)
            model=load(source,entry,cache,[[4,2]],weights=weights,builder_workspace_mb=64,linear_kernel_provider=provider)
            try:
                assert model.runtime_status()['forbidden_path_counters']['graph_cache_hits']==1
                actual=garnet.tensor_to_cpu(execute(model,[input])).tolist()
            finally:model.release_runtime()
            assert actual==expected,(provider,'serialized FP32 output reload',actual)
print('FP32 accumulation output passed native/streamed/registered bias-cancellation and raw-output checks; legacy rounding retained',flush=True)
