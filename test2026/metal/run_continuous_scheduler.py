"""Native scheduler + real mixed-layer GPU execution, compared per request.

Arguments: request-row fixture, native build, Garnet source. One-token prompts
exercise refill/mixed ticks; learned models and chunked prefill are later gates.
"""
import sys,json,math
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
assert 'native_continuous_scheduler' in garnet.backend_info()
from xModel.qwen35.vl_0_8b.decoder_layer import T,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];batch=f['batch'];width=c['hidden_size'];capacity=f['capacity']
shapes=[[[batch,width],[batch,29,4],[batch,3,7,5],[batch]],
        [[batch,width],[batch,1,capacity,6],[batch,1,capacity,6],[batch,3],[batch],[batch]]]
graphs=[];head=None;session=garnet.decoder_session();policy=garnet.continuous_scheduler();references={}
def snapshot(owner):return [[value.tolist() for value in owner.snapshot(layer)] for layer in range(2)]
def near(a,b):
    assert len(a)==len(b)
    for x,y in zip(a,b):assert math.isfinite(x) and abs(x-y)<=5e-6+5e-5*abs(y),(x,y)
def compare_row(actual,wanted,row):
    for layer in range(2):
        for part in range(2):
            n=len(actual[layer][part])//batch
            near(actual[layer][part][row*n:(row+1)*n],wanted[layer][part][row*n:(row+1)*n])
def controls(plan):
    values=[0.0]*(batch*width);positions=[0.0]*(batch*3);slots=[0.0]*batch
    for request in plan['requests']:
        row=request['row'];token=10+request['id'] if plan['stage']=='prefill' else request['token_id']
        for column in range(width):values[row*width+column]=float((token*13+column*7)%29-14)/31
        slots[row]=float(request['slot'])
        for axis in range(3):positions[row*3+axis]=slots[row]
    return (tensor.tensor(values,shape=[batch,width],dtype=tensor.float32),
            tensor.tensor(positions,shape=[batch,3],dtype=tensor.float32),
            tensor.tensor(slots,shape=[batch],dtype=tensor.float32),
            tensor.tensor([float(v) for v in plan['active_rows']],shape=[batch],dtype=tensor.float32))
def run(plan,stop=False,complete=True):
    hidden,positions,slots,active=controls(plan)
    predictions=session.run_leased(graphs,hidden,positions,slots,active,0,plan['epochs'])
    if 'row_final_hidden' in garnet.backend_info():
        bank=session.predict_rows(plan['epochs'])
        for row in range(batch):assert bank[row]==(predictions[row] if plan['active_rows'][row] else -1)
    actual=snapshot(session);actual_hidden=session.hidden().tolist()
    for request in plan['requests']:
        identity=request['id'];row=request['row']
        if identity not in references:
            reference=garnet.decoder_session();reference.build(graphs,head);references[identity]=reference
        reference=references[identity]
        only=dict(plan);only['requests']=[request];only['active_rows']=[1 if i==row else 0 for i in range(batch)]
        h,p,s,a=controls(only);expected=reference.run(graphs,h,p,s,a,0)
        assert expected[row]==predictions[row]
        compare_row(actual,snapshot(reference),row)
        near(actual_hidden[row*width:(row+1)*width],reference.hidden().tolist()[row*width:(row+1)*width])
    stages=[]
    if complete:
        for request in plan['requests']:
            fn=policy.complete_prefill if plan['stage']=='prefill' else policy.complete_decode
            stages.append(fn(request['id'],predictions[request['row']],stop))
    return stages
def reject_old(plan):
    saved=snapshot(session);h,p,s,a=controls(plan);rejected=False
    try:session.run_leased(graphs,h,p,s,a,0,plan['epochs'])
    except Exception:rejected=True
    assert rejected and snapshot(session)==saved
def next_tick():
    result=json.loads(policy.next_json());batches=result['batches'];packed=result['packed']
    assert packed['execution']=='metadata_only'
    expected=[(stage,request) for stage in ('decode','prefill')
        for plan in batches if plan['stage']==stage for request in plan['requests']]
    assert len(packed['segments'])==len(expected)
    offset=0
    for segment,(stage,request) in zip(packed['segments'],expected):
        count=1 if stage=='decode' else request['prefill_tokens']
        assert segment==dict(id=request['id'],row=request['row'],epoch=request['epoch'],
            stage=stage,packed_offset=offset,token_count=count,slot=request['slot'],
            absolute_position=request['slot']+request['position_delta'])
        offset+=count
    assert offset==packed['token_count']
    return batches
try:
    for layer in range(2):
        symbols=[tensor.input('scheduler_'+str(layer)+'_'+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes[layer])]
        if layer==0:
            @T.fusion(name='scheduler_linear',boundary='required')
            def forward(hidden,conv,state,active):return Qwen35LinearDecoderLayer(hidden,conv,state,active,c,0)
        else:
            @T.fusion(name='scheduler_full',boundary='required')
            def forward(hidden,key,value,positions,slots,active):return Qwen35FullDecoderLayer(hidden,key,value,positions,slots,active,c,1,capacity)
        graph=garnet.compiled_graph();graph.build(forward(*symbols),symbols,f['checkpoint']);graphs.append(graph)
    symbol=tensor.input('scheduler_head',shape=[batch,width],dtype=tensor.float32)
    @T.fusion(name='scheduler_head_prediction',boundary='required')
    def predict(hidden):return hidden * T.binary_op('add') * hidden
    head=garnet.compiled_graph();head.build(predict(symbol),[symbol],f['checkpoint'])
    session.build(graphs,head);assert policy.initialize(session,16)
    assert policy.submit(8,1,0)==0 and policy.submit(1,8,0)==0
    ids=[policy.submit(1,2 if i==0 else 4,i) for i in range(4)]
    e=policy.submit(1,3,4);assert e>0
    tick=next_tick();assert len(tick)==1 and tick[0]['stage']=='prefill' and len(tick[0]['requests'])==4
    initial={r['id']:(r['row'],r['epoch']) for r in tick[0]['requests']}
    assert json.loads(policy.status_json())['queued']==1
    rejected=False
    try:next_tick()
    except Exception:rejected=True
    assert rejected and run(tick[0])==[2]*4
    old=next_tick()[0];assert old['stage']=='decode' and len(old['requests'])==4
    assert sorted(run(old))==[2,2,2,3]
    tick=next_tick();assert [p['stage'] for p in tick]==['decode','prefill']
    assert len(tick[0]['requests'])==3 and tick[1]['requests'][0]['id']==e
    replacement=tick[1]['requests'][0]
    assert replacement['row']==initial[ids[0]][0] and replacement['epoch']!=initial[ids[0]][1]
    for request in tick[0]['requests']:assert (request['row'],request['epoch'])==initial[request['id']]
    reject_old(old)
    assert run(tick[0])==[2]*3 and run(tick[1])==[2]
    for _ in range(4):
        tick=next_tick()
        if not tick:break
        for plan in tick:run(plan)
    assert not next_tick() and json.loads(policy.status_json())['active']==0
    assert json.loads(policy.status_json())['tracked']==0
    one=policy.submit(1,1,0);tick=next_tick();assert run(tick[0])==[3] and not next_tick()
    eos=policy.submit(1,4,0);tick=next_tick();assert run(tick[0],True)==[3] and not next_tick()
    queued=policy.submit(1,3,0);assert policy.cancel(queued) and not next_tick()
    cancelled=policy.submit(1,3,0);stale=next_tick()[0];assert policy.cancel(cancelled)
    successor=policy.submit(1,3,0);current=next_tick()[0]
    assert policy.complete_prefill(cancelled,0,False)==-1
    assert policy.complete_prefill(successor,-1,False)==-1
    reject_old(stale);assert run(current)==[2] and policy.cancel(successor)
    closing=policy.submit(1,3,0);pending=next_tick()[0];run(pending,complete=False)
    assert policy.close()
    assert all(value==0 for parts in snapshot(session) for values in parts for value in values)
    reject_old(pending)
    assert session.claim_request_row(999)[0]==0
    print('native_continuous_scheduler PASS combined vs per-request GPU reference, A-to-E mixed ticks, priority, budget/EOS, cancellation, stale dispatch, close and row reuse',flush=True)
finally:
    policy.close();session.close()
    for reference in references.values():reference.close()
    if head is not None:head.close()
    for graph in graphs:graph.close()
    garnet.shutdown()
