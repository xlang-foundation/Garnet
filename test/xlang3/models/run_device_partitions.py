# SPDX-License-Identifier: Apache-2.0
"""Native graph placement, transfer edges, cache reload and output ownership."""
import os,sys,json
import garnet,tensor
root=os.path.abspath(sys.argv[1]);os.makedirs(root,exist_ok=True)
devices=[int(x) for x in sys.argv[2].split(',')]
assert len(devices)==2 and devices[0]!=devices[1]
source=os.path.join(os.path.dirname(os.path.abspath(__file__)),'device_partition_fixture.py')
original=garnet.cuda_set_device(devices[0])
pool=garnet.cuda_engine_pool();pool.enter()
try:
    for reverse in (False,True):
        first,last=devices[::-1] if reverse else devices
        placement=[first,first,last,last]
        cache=os.path.join(root,'reverse' if reverse else 'forward')
        for reload in range(3):
            model=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',cache_dir=cache,
                backend='tensorrt',precision='bf16',input_shapes=[[2],[2]],input_dtypes=['float32']*2,
                compile={'builder_optimization_level':0,'partition':{'devices':placement},'prepare_async':reload==2})
            status=model.runtime_status();assert status['ready'],status
            if reload==2:assert status['engine_preparation_pending'],status
            plan=status['execution_plan'];assert [p['device_id'] for p in plan['engine_partitions']]==placement,plan
            if reload:assert status['forbidden_path_counters']['graph_cache_hits']==1,status
            held=None
            for values in (([1,2],[3,4]),([4,5],[2,3])):
                a,b=values
                inputs=[garnet.tensor_to_device(tensor.tensor(a,dtype=tensor.float32),last),
                    garnet.tensor_to_device(tensor.tensor(b,dtype=tensor.float32),first)]
                result=model.forward({'inputs':inputs})
                assert result['status']=='ok',result
                expected=[((x+y)**2+y)*y for x,y in zip(a,b)]
                assert garnet.tensor_to_cpu(result['output']).tolist()==expected,result
                # A device scope must restore its caller even when the output
                # lives on a different GPU.
                assert garnet.cuda_set_device(devices[0])==devices[0]
                if held is None:held=result['output'];held_values=expected
            model.release_runtime()
            assert garnet.tensor_to_cpu(held).tolist()==held_values
            assert json.loads(pool.stats_json())['contexts']==0
    invalid=garnet.load_model(source,runtime_mode='compiled_xmodel',entry_function='forward',
        cache_dir=os.path.join(root,'invalid'),backend='tensorrt',input_shapes=[[2],[2]],
        compile={'partition':{'devices':[devices[0],devices[1]]}})
    assert not invalid.runtime_status()['ready']
    assert invalid.runtime_status()['error_code']=='partition_device_placement_invalid',invalid.runtime_status()
    invalid.release_runtime()
finally:
    pool.leave();pool.close();garnet.cuda_set_device(original)
print('Native device partitions passed: both layouts, fresh inputs, transfer edges, eager/async cache reload, output lifetime and invalid placement',flush=True)
