"""Default-allocation control for a sanitizer failure in threaded D2D copies."""
import json,os,struct,sys,threading
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T
fixture=json.load(open(sys.argv[1]));output=sys.argv[2];checkpoint=fixture['checkpoint'];h=fixture['config']['hidden_size']
io=garnet.cuda_tensor_io();io.open(checkpoint)
names=['model.language_model.layers.0.mlp.'+kind+'.weight' for kind in ('gate_proj','up_proj','down_proj')]
weights=[io.weight(name) for name in names];graphs=[];hidden=[io.zeros([16,h]),io.zeros([16,h])]
values=[[float((i*17+repeat*11)%63-31)/64 for i in range(16*h)] for repeat in range(12)]
expected=[[],[]];copies=[[],[]];failures=[];barrier=threading.Barrier(2)
def bits(value):
    data=garnet.tensor_to_cpu(value).tolist();return struct.pack('<'+str(len(data))+'f',*data)
def build():
    symbols=[tensor.input('thread_lifetime_hidden',shape=[16,h],dtype=tensor.float32)]
    for index,name in enumerate(names):symbols.append(tensor.input('thread_lifetime_w'+str(index),shape=json.loads(io.weight_info(name))['shape'],dtype=tensor.bfloat16))
    @T.fusion(name='default_output_lifetime_probe',boundary='required')
    def forward(x,*bound):return x*T.unary_op('qwen35_mlp',weight_prefix='model.language_model.layers.0.mlp',external_weights=dict(zip(names,bound)))
    graph=garnet.compiled_graph();graph.build(forward(*symbols),symbols,checkpoint)
    graph.enable_cuda_graph(True);graph.enable_serial_graph_reuse(True);graphs.append(graph)
try:
    build();build()
    for index,graph in enumerate(graphs):
        assert not json.loads(graph.stats_json())['user_managed_workspace']
        for value in values:
            io.update(hidden[index],value);expected[index].append(bits(graph.run_device([hidden[index]]+weights)))
    print('DEFAULT_OUTPUT_LIFETIME_CONTROL_READY',flush=True)
    def worker(index):
        try:
            for value in values:
                barrier.wait(timeout=60);io.update(hidden[index],value)
                copies[index].append(io.clone(graphs[index].run_device([hidden[index]]+weights)))
        except BaseException as error:failures.append(str(error));barrier.abort()
    threads=[threading.Thread(target=worker,args=(index,)) for index in range(2)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    assert not failures,failures
    print('DEFAULT_OUTPUT_LIFETIME_THREADS_JOINED',flush=True)
    for index in range(2):
        assert len(copies[index])==12
        for repeat,value in enumerate(copies[index]):assert bits(value)==expected[index][repeat]
    print('DEFAULT_OUTPUT_LIFETIME_ALL_CLONES_OBSERVED',flush=True)
    for graph in graphs:graph.close()
    io.close()
    with open(output,'w') as stream:json.dump({'all_output_checks_passed':True,'output_checks':24,'user_managed_workspace':False,
        'scope':'Default-workspace control only. Sanitizer exit/log, including runtime teardown, determines memory safety.'},stream,indent=2)
    print('DEFAULT_OUTPUT_LIFETIME_BODY_COMPLETE',flush=True)
finally:
    for graph in graphs:graph.close()
    io.close()
