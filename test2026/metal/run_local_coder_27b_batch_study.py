"""Learned batch2 native scheduling and variable-length prompt correctness.

XLang arguments: build, source, checkpoint folder, frozen study JSON, result.
Serial masked prefill avoids assuming equal prompt lengths or a shared final
head index. Decode is genuinely combined. Not a performance campaign.
"""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen38.vl_27b.metal_batch_study import create_decoder
folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[4]) as stream:study=json.load(stream)
cases=study['requests'];batch=2;chunk=8;width=config['text_config']['hidden_size']
assert len(cases)==batch
capacity=study['capacity'];limit=study['output_tokens']
assert type(limit) is int and 2<=limit<=32
assert all(len(case['expected_ids'])==limit and len(case['token_ids'])+limit<=capacity for case in cases)
decoder=None;policy=None
report=dict(status='running',scope=__doc__,capacity=capacity,batch=batch,prefill_chunk=chunk,
            requested_output_tokens=limit,backend=garnet.backend_info(),runs=[])
def save():
    with open(sys.argv[5],'w') as stream:stream.write(json.dumps(report,indent=2)+'\n')
def tensor_f32(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
try:
    report['memory_before']=json.loads(garnet.memory_stats_json());start=time.monotonic()
    decoder=create_decoder(folder+'/model.safetensors.index.json',config,capacity)
    report['initialization_seconds']=time.monotonic()-start
    report['memory_after_decoder']=json.loads(garnet.memory_stats_json());save()
    decoder.prefill_compile_ms=0
    prefill_graphs=decoder._resident_prefill_graphs(chunk)
    report['memory_after_prefill_compile']=json.loads(garnet.memory_stats_json());save()
    for repeat in range(2):
        policy=garnet.continuous_scheduler();policy.initialize(decoder.session,8)
        ids=[policy.submit(len(case['token_ids']),limit,0) for case in cases]
        assert all(identity>0 for identity in ids)
        tick=json.loads(policy.next_json())['batches'];assert len(tick)==1 and tick[0]['stage']=='prefill'
        plan=tick[0];generated={identity:[] for identity in ids};began=time.monotonic()
        for request in plan['requests']:
            identity=request['id'];row=request['row'];tokens=cases[ids.index(identity)]['token_ids']
            offset=0
            while offset<len(tokens):
                length=min(chunk,len(tokens)-offset);values=[]
                for t in range(chunk):
                    values.append(decoder.io.embedding([tokens[offset+t] if r==row else 0 for r in range(batch)]) if t<length else decoder.io.zeros([batch,width]))
                hidden=decoder.io.stack_tokens(values)
                flags=[1.0 if r==row and t<length else 0.0 for r in range(batch) for t in range(chunk)]
                slots=[float(offset+t) if r==row and t<length else 0.0 for r in range(batch) for t in range(chunk)]
                positions=[float(offset+t) if r==row and t<length else 0.0 for r in range(batch) for t in range(chunk) for axis in range(3)]
                final=offset+length==len(tokens)
                predicted=decoder.session.run_leased(prefill_graphs,hidden,tensor_f32(positions,[batch,chunk,3]),
                    tensor_f32(slots,[batch,chunk]),tensor_f32(flags,[batch,chunk]),length-1 if final else -1,plan['epochs'])
                if final:
                    generated[identity].append(predicted[row]);assert policy.complete_prefill(identity,predicted[row],False)==2
                offset+=length
        prefill_seconds=time.monotonic()-began;decode_start=time.monotonic()
        for index in range(1,limit):
            tick=json.loads(policy.next_json())['batches'];assert len(tick)==1 and tick[0]['stage']=='decode'
            plan=tick[0];assert len(plan['requests'])==batch
            token_ids=[0]*batch;slots=[0.0]*batch
            for request in plan['requests']:token_ids[request['row']]=request['token_id'];slots[request['row']]=float(request['slot'])
            positions=[position for slot in slots for position in (slot,slot,slot)]
            predicted=decoder.session.run_leased([layer['graph'] for layer in decoder.layers],decoder.io.embedding(token_ids),
                tensor_f32(positions,[batch,3]),tensor_f32(slots,[batch]),tensor_f32([float(v) for v in plan['active_rows']],[batch]),0,plan['epochs'])
            for request in plan['requests']:
                identity=request['id'];token=predicted[request['row']];generated[identity].append(token)
                assert policy.complete_decode(identity,token,index+1==limit)==(3 if index+1==limit else 2)
        sequences=[generated[identity] for identity in ids]
        assert sequences==[case['expected_ids'] for case in cases],'Batch rows differ from frozen single-request native tokens'
        status=json.loads(policy.status_json());assert status['active']==0 and status['tracked']==0
        report['runs'].append(dict(repeat=repeat,generated_ids=sequences,prefill_seconds=prefill_seconds,
            decode_window_seconds=time.monotonic()-decode_start,memory=json.loads(garnet.memory_stats_json()),scheduler=status))
        assert policy.close();policy=None;save()
    report['status']='batch2_study_passed_not_full_qualification'
except:
    report['status']='failed';raise
finally:
    if policy is not None:policy.close()
    if decoder is not None:
        report['memory_before_close']=json.loads(garnet.memory_stats_json())
        decoder.session.close();report['memory_after_session_close']=json.loads(garnet.memory_stats_json())
        for graph in decoder.prefill_graphs.values():graph.close()
        report['memory_after_prefill_close']=json.loads(garnet.memory_stats_json())
        for layer in decoder.layers:layer['graph'].close()
        report['memory_after_layers_close']=json.loads(garnet.memory_stats_json())
        decoder.close()
    report['memory_after_close']=json.loads(garnet.memory_stats_json());save()
    decoder=None;prefill_graphs=[];hidden=None;values=[];policy=None
    time.sleep(0.5)
    report['memory_after_close_quiescence']=json.loads(garnet.memory_stats_json());save();garnet.shutdown()
