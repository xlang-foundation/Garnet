# SPDX-License-Identifier: Apache-2.0
"""Variable sequence lengths vs independent scalar math, fresh and cached."""
import math,os,sys
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
source=os.path.join(os.path.dirname(os.path.abspath(__file__)),'dynamic_primitives.py')
def load(entry):
    return garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function=entry,backend='tensorrt',
        cache_dir=os.path.join(root,entry),input_shapes=[[1,4,32],[1,4,2,16],[1,4,2,16]],input_dtypes=['float32']*3,
        compile={'builder_optimization_level':0,'input_shape_profiles':{'min':[[1,4,32],[1,4,2,16],[1,4,2,16]],
            'opt':[[1,4,32],[1,4,2,16],[1,4,2,16]],'max':[[1,8,32],[1,8,2,16],[1,8,2,16]],
            'batch_axes':[-1]*3,'batch_units':[1]*3,'sequence_axes':[1]*3}})
def inputs(length,fused):
    values=[(c-16)/32 if fused else .03*(i+1)+.01*(c-11) for i in range(length) for c in range(32)]
    args=[tensor.tensor(values,shape=[1,length,32],dtype=tensor.float32)]
    args += [tensor.tensor([v]*(length*32),shape=[1,length,2,16],dtype=tensor.float32) for v in (1.0,0.0)]
    return values,args
def reference(values,length):
    normalized=[]
    for i in range(length):
        heads=[]
        for h in range(2):
            row=values[i*32+h*16:i*32+h*16+16];scale=math.sqrt(sum(v*v for v in row)/16+1e-6)
            heads.append([v/scale for v in row])
        normalized.append(heads)
    output=[]
    for i in range(length):
        for h in range(2):
            scores=[sum(a*b for a,b in zip(normalized[i][h],normalized[j][h]))/4 for j in range(length)]
            maximum=max(scores);weights=[math.exp(s-maximum) for s in scores];total=sum(weights)
            row=[sum(weights[j]*normalized[j][h][c] for j in range(length))/total for c in range(16)]
            mean=sum(row)/16;scale=math.sqrt(sum((v-mean)**2 for v in row)/16+1e-6)
            output += [(v-mean)/scale for v in row]
    return output
for entry in ('forward','fused'):
    for reload in (False,True):
        model=load(entry);status=model.runtime_status();assert status['ready'],status
        if reload:assert status['forbidden_path_counters']['graph_cache_hits']==1,status
        held=None
        for length in (4,8,6,4):
            values,args=inputs(length,entry=='fused');result=model.forward({'inputs':args});assert result['status']=='ok',result
            actual=garnet.tensor_to_cpu(result['output']).tolist();expected=values if entry=='fused' else reference(values,length)
            # TensorRT permits TF32 by default. A fixed length-8 graph also
            # differs from scalar FP64 math by 6.73e-4 on this fixture.
            error=max(abs(a-b) for a,b in zip(actual,expected));assert error<0.002,(entry,length,error)
            if entry=='forward' and not reload:
                fixed=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='fixed',backend='tensorrt',
                    cache_dir=os.path.join(root,'fixed-'+str(length)),input_shapes=[[1,length,32],[1,length,2,16],[1,length,2,16]],
                    input_dtypes=['float32']*3,compile={'builder_optimization_level':0})
                assert fixed.runtime_status()['ready'],fixed.runtime_status()
                fixed_result=fixed.forward({'inputs':args});assert fixed_result['status']=='ok',fixed_result
                fixed_values=garnet.tensor_to_cpu(fixed_result['output']).tolist()
                assert max(abs(a-b) for a,b in zip(fixed_values,expected))<0.002
                assert max(abs(a-b) for a,b in zip(fixed_values,actual))<0.002
                fixed.release_runtime()
            assert list(result['output'].shape)==[1,length,32]
            if held is None:held=result['output'];saved=actual
        for length in (3,9):
            values,args=inputs(length,entry=='fused');assert model.forward({'inputs':args})['status']=='error'
        model.release_runtime();assert garnet.tensor_to_cpu(held).tolist()==saved
invalid=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='invalid',backend='tensorrt',
    cache_dir=os.path.join(root,'invalid'),input_shapes=[[1,4,32]],input_dtypes=['float32'])
assert not invalid.runtime_status()['ready'],invalid.runtime_status()
invalid.release_runtime()
print('Dynamic primitives passed: inferred reshape, slice/concat rotary, RMS/LayerNorm, F32 and fused BF16 attention, interior shapes, cache reuse, output lifetime and bounds',flush=True)
