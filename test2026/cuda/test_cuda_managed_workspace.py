"""Garnet SDK workspace binding, ordering, aliases and retained ownership."""
import json,os,struct,sys,threading
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];checkpoint=fixture['checkpoint'];h=fixture['config']['hidden_size']
io=garnet.cuda_tensor_io();io.open(checkpoint);graphs=[];rejections=[]
names=['model.language_model.layers.0.mlp.'+kind+'.weight' for kind in ('gate_proj','up_proj','down_proj')]
weights=[io.weight(name) for name in names]

def bits(value):
    values=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(values))+'f',*values)

def reject(label,action):
    try:action()
    except BaseException as error:rejections.append({'case':label,'reason':str(error)});return
    raise AssertionError('Invalid workspace action accepted: '+label)

def build(rows,managed):
    x=tensor.input('workspace_hidden',shape=[rows,h],dtype=tensor.float32);symbols=[x]
    for i,name in enumerate(names):
        info=json.loads(io.weight_info(name));symbols.append(tensor.input('workspace_weight_'+str(i),shape=info['shape'],dtype=tensor.bfloat16))
    @T.fusion(name='qwen35_managed_workspace_probe',boundary='required')
    def forward(hidden,*bound):return hidden*T.unary_op('qwen35_mlp',weight_prefix='model.language_model.layers.0.mlp',external_weights=dict(zip(names,bound)))
    graph=garnet.compiled_graph()
    try:
        graph.enable_user_managed_workspace(managed);graph.build(forward(*symbols),symbols,checkpoint)
        graph.enable_cuda_graph(True);graph.enable_serial_graph_reuse(True)
    except:graph.close();raise
    graphs.append(graph);return graph

try:
    shapes=(4,16);control=[build(rows,False) for rows in shapes];candidate=[build(rows,True) for rows in shapes]
    requirements=[json.loads(g.stats_json())['context_workspace_required_bytes'] for g in candidate]
    assert min(requirements)>4,'Probe must exercise actual nonzero context scratch'
    words=(max(requirements)+3)//4;large=io.zeros([words+64]);small=io.storage_view(large,[1]);pool=io.zeros([words])
    hidden=[io.zeros([rows,h]) for rows in shapes]
    reject('bind unbuilt',lambda:garnet.compiled_graph().set_workspace_storage(pool))
    reject('disabled policy',lambda:control[0].set_workspace_storage(pool))
    reject('policy change after build',lambda:candidate[0].enable_user_managed_workspace(False))
    reject('run before binding',lambda:candidate[0].run_device([hidden[0]]+weights))
    reject('logical view too small despite large backing span',lambda:candidate[0].set_workspace_storage(small))
    reject('host storage',lambda:candidate[0].set_workspace_storage(garnet.tensor_to_cpu(pool)))
    reject('symbolic storage',lambda:candidate[0].set_workspace_storage(tensor.input('unmaterialized_workspace',shape=[words],dtype=tensor.float32)))
    reject('readonly BF16 checkpoint storage',lambda:candidate[0].set_workspace_storage(weights[0]))
    for graph in candidate:graph.set_workspace_storage(pool)
    reject('second binding',lambda:candidate[0].set_workspace_storage(pool))
    reject('host run',lambda:candidate[0].run([hidden[0]]+weights))
    aliased=io.storage_view(pool,[shapes[0],h])
    reject('input overlaps workspace',lambda:candidate[0].run_device([aliased]+weights))
    bad_output=build(shapes[0],True);bad_output.set_workspace_storage(pool);bad_output.set_output_storage(aliased)
    reject('output overlaps workspace',lambda:bad_output.run_device([hidden[0]]+weights))
    assert json.loads(bad_output.stats_json())['graph_captures']==0
    bad_output.close()
    expected=[[],[]];values=[[],[]]
    for index,rows in enumerate(shapes):
        for repeat in range(12):
            value=[float((i*17+repeat*11)%63-31)/64 for i in range(rows*h)];values[index].append(value)
            io.update(hidden[index],value);expected[index].append(bits(control[index].run_device([hidden[index]]+weights)))
    checks=0
    for repeat in range(12):
        for index,graph in enumerate(candidate):
            io.update(hidden[index],values[index][repeat]);assert bits(graph.run_device([hidden[index]]+weights))==expected[index][repeat];checks+=1
    # Enqueue clones before the next call reuses each graph's private output.
    # CPU observation occurs only after both host threads have submitted.
    copies=[[],[]];failures=[];barrier=threading.Barrier(2)
    def worker(index):
        try:
            for repeat in range(12):
                barrier.wait(timeout=60);io.update(hidden[index],values[index][repeat])
                copies[index].append(io.clone(candidate[index].run_device([hidden[index]]+weights)))
        except BaseException as error:
            failures.append(str(error));barrier.abort()
    threads=[threading.Thread(target=worker,args=(index,)) for index in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert not failures,failures
    for index in range(2):
        assert len(copies[index])==12
        for repeat,value in enumerate(copies[index]):assert bits(value)==expected[index][repeat];checks+=1
    stats=[json.loads(graph.stats_json()) for graph in candidate]
    assert all(s['user_managed_workspace'] and s['external_workspace_bound'] and s['external_workspace_capacity_bytes']==words*4 for s in stats)
    assert all(s['contexts']==1 and s['graph_replays']>0 and s['graph_capture_failures']==0 for s in stats)
    for graph in control:graph.close()
    del pool,large,small,aliased
    candidate[0].close();io.update(hidden[1],values[1][-1]);io.close()
    # The remaining graph retains the shared allocation and readonly weights.
    last=candidate[1].run_device([hidden[1]]+weights)
    assert bits(last)==expected[1][-1];checks+=1
    candidate[1].close();assert bits(last)==expected[1][-1];checks+=1
    result={'all_passed':True,'output_checks':checks,'invalid_binding_rejections':rejections,'managed_graphs':stats,
        'pool_capacity_bytes_count_once':words*4,'threaded_cloned_outputs':24,
        'caller_pool_drop_partial_graph_close_io_close_and_last_output_owner':True,
        'scope':'Garnet native/SDK workspace, exact same-program outputs and ownership. No pretrained state, model memory, speed or independent quality acceptance.'}
    with open(output,'w') as stream:json.dump(result,stream,indent=2)
    print('GARNET_MANAGED_WORKSPACE_PASS',checks,flush=True)
finally:
    for graph in graphs:graph.close()
    io.close()
