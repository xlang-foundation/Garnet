"""Native four-row lifecycle against independently computed mixed-state steps."""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity']
shapes=[[[batch,width],[batch,29,4],[batch,3,7,5],[batch]],
        [[batch,width],[batch,1,capacity,6],[batch,1,capacity,6],[batch,3],[batch],[batch]]]
graphs=[];session=garnet.decoder_session();head=None
def close_enough(actual,expected):
    assert len(actual)==len(expected)
    for x,y in zip(actual,expected):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(x,y)
def snapshot():return [[value.tolist() for value in session.snapshot(layer)] for layer in range(2)]
try:
    assert 'request_row_reset' in garnet.backend_info()
    for layer in range(2):
        symbols=[tensor.input('request_row_'+str(layer)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes[layer])]
        if layer==0:
            @T.fusion(name='request_row_linear_reference',boundary='required')
            def forward(hidden,conv,state,active):return Qwen35LinearDecoderLayer(hidden,conv,state,active,c,0)
        else:
            @T.fusion(name='request_row_attention_reference',boundary='required')
            def forward(hidden,key,value,positions,slots,active):return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,1,capacity)
        graph=garnet.compiled_graph();graph.build(forward(*symbols),symbols,f['checkpoint']);graphs.append(graph)
    if f['with_head']:
        head_symbol=tensor.input('request_row_test_head',shape=[batch,width],dtype=tensor.float32)
        @T.fusion(name='request_row_test_prediction',boundary='required')
        def head_forward(hidden):return hidden * T.binary_op('add') * hidden
        head=garnet.compiled_graph();head.build(head_forward(head_symbol),[head_symbol],f['checkpoint'])
    session.build(graphs,head)
    for step in f['steps']:
        if step['reset_row']>=0:
            saved=snapshot();hidden_saved=session.hidden().tolist();row=step['reset_row']
            session.reset_row(row);actual=snapshot()
            for layer in range(2):
                for part in range(2):
                    columns=len(saved[layer][part])//batch
                    for i,value in enumerate(actual[layer][part]):
                        assert value==(0 if i//columns==row else saved[layer][part][i])
            for i,value in enumerate(session.hidden().tolist()):assert value==(0 if i//width==row else hidden_saved[i])
            assert session.row_reset_bytes==(29*4+3*7*5+2*capacity*6+width+(2 if f['with_head'] else 0))*4
        hidden=tensor.tensor(step['hidden'],shape=[batch,width],dtype=tensor.float32)
        positions=tensor.tensor(step['positions'],shape=[batch,3],dtype=tensor.float32)
        slots=tensor.tensor(step['slots'],shape=[batch],dtype=tensor.float32)
        active=tensor.tensor(step['active'],shape=[batch],dtype=tensor.float32)
        predicted=session.run(graphs,hidden,positions,slots,active,0 if f['with_head'] else -1)
        if f['with_head']:assert predicted==step['expected_predictions']
        close_enough(session.hidden().tolist(),step['expected_hidden'])
        actual=snapshot()
        for layer in range(2):
            for part in range(2):close_enough(actual[layer][part],step['expected_states'][layer][part])
    saved=snapshot()
    for row in (-1,batch):
        rejected=False
        try:session.reset_row(row)
        except Exception:rejected=True
        assert rejected and snapshot()==saved
    if 'retired_inplace_defaults' in garnet.backend_info():
        # The optimized session borrows no default KV output storage. Standalone
        # graph.run must rehydrate its defaults and leave private rows untouched.
        assert graphs[1].lazy_default_planned_slots>=5
        step=f['steps'][0]
        hidden=tensor.tensor(step['hidden'],shape=[batch,width],dtype=tensor.float32)
        active=tensor.tensor(step['active'],shape=[batch],dtype=tensor.float32)
        conv=tensor.tensor([0.0]*(batch*29*4),shape=shapes[0][1],dtype=tensor.float32)
        state=tensor.tensor([0.0]*(batch*3*7*5),shape=shapes[0][2],dtype=tensor.float32)
        linear=graphs[0].run([hidden,conv,state,active]).tolist()
        packed=width+29*4+3*7*5
        values=[]
        for row in range(batch):values+=linear[row*packed:row*packed+width]
        hidden=tensor.tensor(values,shape=[batch,width],dtype=tensor.float32)
        key=tensor.tensor([0.0]*(batch*capacity*6),shape=shapes[1][1],dtype=tensor.float32)
        value=tensor.tensor([0.0]*(batch*capacity*6),shape=shapes[1][2],dtype=tensor.float32)
        positions=tensor.tensor(step['positions'],shape=[batch,3],dtype=tensor.float32)
        slots=tensor.tensor(step['slots'],shape=[batch],dtype=tensor.float32)
        full=graphs[1].run([hidden,key,value,positions,slots,active]).tolist()
        packed=width+2*capacity*6
        values=[]
        for row in range(batch):values+=full[row*packed:row*packed+width]
        close_enough(values,step['expected_hidden'])
        assert snapshot()==saved
        session.reset()
        hidden=tensor.tensor(step['hidden'],shape=[batch,width],dtype=tensor.float32)
        session.run(graphs,hidden,positions,slots,active,0 if f['with_head'] else -1)
        close_enough(session.hidden().tolist(),step['expected_hidden'])
        print('retired_inplace_defaults PASS legacy rehydration and resident reuse',flush=True)
    print('native_request_rows PASS batch4 mixed linear/KV independent reference, A-to-E replacement, B/C/D isolation, inactive/reactivated C, bounds, with_head',f['with_head'],flush=True)
finally:
    session.close()
    for graph in graphs:graph.close()
    if head is not None:head.close()
    garnet.shutdown()
