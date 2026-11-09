# SPDX-License-Identifier: Apache-2.0
"""Independent scalar dense chain; native streamed weights and dynamic plans."""
import os,sys,json,struct
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
provider=sys.argv[2] if len(sys.argv)>2 else 'streamed_bf16'
source=os.path.join(os.path.dirname(__file__),'streamed_projection_fixture.py')
weights=os.path.join(root,'weights');os.makedirs(weights,exist_ok=True)
def bf16(value):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    return (bits+32767+((bits>>16)&1))>>16
def rounded(value):return struct.unpack('<f',struct.pack('<I',bf16(value)<<16))[0]
matrices=[[(int((i*7+layer*3)%17)-8)/32 for i in range(256)] for layer in range(4)]
biases=[[(i-layer)/128 for i in range(16)] for layer in range(4)]
header={};body=b''
for layer in range(4):
    for suffix,values,shape in [('weight',matrices[layer],[16,16]),('bias',biases[layer],[16])]:
        data=struct.pack('<'+'H'*len(values),*[bf16(v) for v in values])
        header['layer.'+str(layer)+'.'+suffix]={'dtype':'BF16','shape':shape,'data_offsets':[len(body),len(body)+len(data)]}
        body+=data
encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
with open(os.path.join(weights,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(encoded))+encoded+body)
with open(os.path.join(weights,'config.json'),'w') as f:json.dump({},f)
def load(provider,partitioned):
    options={'builder_optimization_level':0,'input_shape_profiles':{'min':[[1,4,16]],'opt':[[1,4,16]],'max':[[1,8,16]],'batch_axes':[-1],'batch_units':[1],'sequence_axes':[1]}}
    if provider:options['linear_kernel_provider']=provider
    if partitioned:options['partition']={'max_atomic_regions_per_partition':1}
    return garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',weights=weights,
        cache_dir=os.path.join(root,('partitioned-' if partitioned else 'single-')+(str(provider) if provider else 'default')),backend='tensorrt',
        input_shapes=[[1,4,16]],input_dtypes=['float32'],compile=options)
def reference(values,length):
    for layer in range(4):
        values=[rounded(sum(rounded(values[row*16+k])*matrices[layer][c*16+k] for k in range(16)))+biases[layer][c] for row in range(length) for c in range(16)]
    return values
for partitioned in (False,True):
    for repeat in (False,True):
        pool=garnet.cuda_engine_pool();pool.enter()
        model=load(provider,partitioned)
        try:
            status=model.runtime_status();assert status['ready'],status
            assert status['execution_plan']['planner_options']['linear_kernel_provider']==provider,status
            if repeat:assert status['forbidden_path_counters']['graph_cache_hits']==1,status
            assert json.loads(status['streamed_projection_stats_json'])['active_device_pools']==0,status
            held=None
            for length in (4,8,6,4):
                values=[(int((i*5+int(repeat))%23)-11)/64 for i in range(length*16)]
                result=model.forward({'inputs':[tensor.tensor(values,shape=[1,length,16],dtype=tensor.float32)]})
                assert result['status']=='ok',result
                actual=garnet.tensor_to_cpu(result['output']).tolist();expected=reference(values,length)
                assert list(result['output'].shape)==[1,length,16]
                assert actual==expected,(partitioned,length,max(abs(a-b) for a,b in zip(actual,expected)))
                stats=json.loads(model.runtime_status()['streamed_projection_stats_json'])
                assert stats['active_device_pools']==1 and stats['mapped_files']==1,stats
                assert stats['pinned_staging_bytes']==(512<<20) and stats['device_staging_bytes']==(512<<20),stats
                if provider=='registered_bf16':assert stats['registered_checkpoint_bytes']>0,stats
                if provider=='registered_bf16' and int(os.environ.get('GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB','0'))>0:
                    assert stats['device_cached_weight_bytes']>0,stats
                if held is None:held=result['output'];saved=actual
            model.release_runtime();assert garnet.tensor_to_cpu(held).tolist()==saved
            if provider=='registered_bf16':assert json.loads(pool.stats_json())['external_projection_staging']['device_cached_weight_bytes']==0
            assert json.loads(pool.stats_json())['contexts']==0
        finally:model.release_runtime();pool.leave();pool.close()
        after=json.loads(pool.stats_json())['external_projection_staging']
        assert after['active_device_pools']==0 and after['mapped_files']==0 and after['registered_checkpoint_bytes']==0,after
if provider=='registered_bf16' and os.environ.get('GARNET_CUDA_PROJECTION_DEVICE_CACHE_MIB')=='1':
    # Four 512 KiB matrices exceed a 1 MiB device cache. A diagonal chain gives
    # an independent exact answer while exercising retained and streamed weights.
    directory=os.path.join(root,'cache-budget-weights');os.makedirs(directory,exist_ok=True)
    width=512;header={};body=b''
    matrix=[0]*(width*width)
    for i in range(width):matrix[i*width+i]=bf16(0.5)
    for layer in range(4):
        for suffix,raw,shape in [('weight',struct.pack('<'+'H'*len(matrix),*matrix),[width,width]),('bias',b'\x00'*(width*2),[width])]:
            header['layer.'+str(layer)+'.'+suffix]={'dtype':'BF16','shape':shape,'data_offsets':[len(body),len(body)+len(raw)]};body+=raw
    encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
    with open(os.path.join(directory,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(encoded))+encoded+body)
    with open(os.path.join(directory,'config.json'),'w') as f:json.dump({},f)
    model=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',weights=directory,
        cache_dir=os.path.join(root,'cache-budget-engines'),backend='tensorrt',input_shapes=[[1,4,width]],input_dtypes=['float32'],
        compile={'builder_optimization_level':0,'linear_kernel_provider':provider})
    try:
        assert model.runtime_status()['ready'],model.runtime_status()
        previous=None
        for iteration in range(2):
            values=[(i%17-8)/64 for i in range(4*width)]
            result=model.forward({'inputs':[tensor.tensor(values,shape=[1,4,width],dtype=tensor.float32)]})
            assert result['status']=='ok',result
            assert garnet.tensor_to_cpu(result['output']).tolist()==[v/16 for v in values]
            stats=json.loads(model.runtime_status()['streamed_projection_stats_json'])
            assert stats['device_cached_weight_bytes']==(1<<20),stats
            if previous is not None:
                assert stats['device_cache_hits']>=previous['device_cache_hits']+2,stats
                assert stats['uploaded_weight_bytes']-previous['uploaded_weight_bytes']==(1<<20),stats
            previous=stats
    finally:model.release_runtime()
for dtype,inner,entry in [('BF16',16400,'projection'),('F32',16,'projection'),('BF16',16,'projection_fp32'),('BF16',16,'projection_accumulation_fp32')]:
    fallback=os.path.join(root,'fallback-'+dtype+'-'+entry);os.makedirs(fallback,exist_ok=True)
    matrix=[0.0]*(16*inner)
    for column in range(16):matrix[column*inner+column]=0.5
    values=[(column-8)/128 for column in range(16)]
    header={};body=b''
    for suffix,items,shape in [('weight',matrix,[16,inner]),('bias',values,[16])]:
        raw=struct.pack('<'+'H'*len(items),*[bf16(v) for v in items]) if dtype=='BF16' else struct.pack('<'+'f'*len(items),*items)
        header['layer.0.'+suffix]={'dtype':dtype,'shape':shape,'data_offsets':[len(body),len(body)+len(raw)]};body+=raw
    encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
    with open(os.path.join(fallback,'model.safetensors'),'wb') as f:f.write(struct.pack('<Q',len(encoded))+encoded+body)
    with open(os.path.join(fallback,'config.json'),'w') as f:json.dump({},f)
    for repeat in (False,True):
        model=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function=entry,weights=fallback,
            cache_dir=os.path.join(root,'cache-fallback-'+dtype+'-'+entry),backend='tensorrt',input_shapes=[[1,1,inner]],input_dtypes=['float32'],
            compile={'builder_optimization_level':0,'linear_kernel_provider':provider})
        try:
            status=model.runtime_status();assert status['ready'],status
            if repeat:assert status['forbidden_path_counters']['graph_cache_hits']==1,status
            result=model.forward({'inputs':[tensor.tensor([0.25]*inner,shape=[1,1,inner],dtype=tensor.float32)]})
            assert result['status']=='ok',result
            assert garnet.tensor_to_cpu(result['output']).tolist()==[0.125+v for v in values],dtype
            assert json.loads(model.runtime_status()['streamed_projection_stats_json'])['active_device_pools']==0,dtype
        finally:model.release_runtime()
for invalid in ('unknown_provider',True):
    rejected=False
    try:
        model=load(invalid,False)
        rejected=not model.runtime_status()['ready'];model.release_runtime()
    except Exception:rejected=True
    assert rejected,invalid
print('Streamed BF16 projections passed: scalar chains, dynamic shapes, single/partitioned reload, bounded BF16/F32 and explicit FP32 fallback, invalid policy rejection and cleanup',flush=True)
