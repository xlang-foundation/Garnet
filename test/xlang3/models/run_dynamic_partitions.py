# SPDX-License-Identifier: Apache-2.0
"""Weighted partition profiles, changing boundary extents and fresh inputs."""
import os,sys,json,struct
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
devices=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else None
source=os.path.join(os.path.dirname(os.path.abspath(__file__)),'dynamic_partition_fixture.py')
weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
matrix=[((i*7)%17-8)/8 for i in range(64)];raw=struct.pack('<'+'f'*64,*matrix)
header=json.dumps({'proj.weight':{'dtype':'F32','shape':[8,8],'data_offsets':[0,len(raw)]}}).encode();header+=b' '*((-len(header))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(header))+header+raw)
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
def load(asynchronous=False,maximum=8):
    options={'builder_optimization_level':0,'prepare_async':asynchronous,
        'input_shape_profiles':{'min':[[1,4,8],[1,4,8],[1,1,8]],'opt':[[1,4,8],[1,4,8],[1,1,8]],
            'max':[[1,maximum,8],[1,maximum,8],[1,1,8]],'batch_axes':[-1]*3,'batch_units':[1]*3,'sequence_axes':[1,1,-1]}}
    if devices is not None:options['partition']={'devices':[devices[0],devices[0],devices[1],devices[1]]}
    return garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',weights=weights,
        cache_dir=os.path.join(root,'wide' if maximum>8 else 'cache'),backend='tensorrt',
        input_shapes=[[1,4,8],[1,4,8],[1,1,8]],input_dtypes=['float32']*3,compile=options)
pool=garnet.cuda_engine_pool();pool.enter()
try:
    for run in range(3):
        model=load(run==2);status=model.runtime_status();assert status['ready'],status
        parts=status['execution_plan']['engine_partitions'];assert len(parts)==4,parts
        assert any(any(r['min'][1]==5 and r['max'][1]==9 for r in p['input_shape_profiles']) for p in parts),parts
        if run:assert status['forbidden_path_counters']['graph_cache_hits']==1,status
        held=None
        for length in (4,8,6,4):
            a=[((i*3+run)%17-8)/8 for i in range(length*8)];b=[((i*5)%11-5)/8 for i in range(length*8)];c=[i/8 for i in range(8)]
            expected=[2*(sum((a[t*8+i]+b[t*8+i])*matrix[row*8+i] for i in range(8))+b[t*8+row]) for t in range(length) for row in range(8)]+[2*v for v in c]
            inputs=[tensor.tensor(values,shape=shape,dtype=tensor.float32) for values,shape in [(a,[1,length,8]),(b,[1,length,8]),(c,[1,1,8])]]
            result=model.forward({'inputs':inputs});assert result['status']=='ok',result
            assert list(result['output'].shape)==[1,length+1,8]
            actual=garnet.tensor_to_cpu(result['output']).tolist();assert actual==expected,(length,actual,expected)
            if held is None:held=result['output'];saved=actual
        model.release_runtime();assert garnet.tensor_to_cpu(held).tolist()==saved
        assert json.loads(pool.stats_json())['contexts']==0
    wide=load(maximum=20000);status=wide.runtime_status();assert not status['ready'] and '16384' in status['error_message'],status
    wide.release_runtime()
finally:pool.leave();pool.close()
print('Dynamic partitions passed: weighted four-part graph, S+1 boundaries, interior sizes, cache reload, async loading, output lifetime and profile limit',flush=True)
