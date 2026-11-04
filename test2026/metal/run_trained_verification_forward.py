"""Trained-target forward parity, not speculative decoding or a speed benchmark.

XLang arguments: fixture JSON, native build directory, Garnet source root.
The controller must rehash pinned assets and run this engine in isolation.
"""
import json,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
with open(f['config']) as stream:c=json.load(stream)['text_config']
width=c['hidden_size'];capacity=f['capacity'];chunk=4;batch=1
assert len(c['layer_types'])==64 and width==5120
assert 'compiled_verification_fp32' in garnet.backend_info()
base=[];packed=[];heads=[];session=garnet.decoder_session();reference=garnet.decoder_session();io=garnet.tensor_io()
records=[]
def syms(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def value(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def fingerprint(owner):
    result=[]
    for i in range(64):result.append([io.verification_f32_sha256(part) for part in owner.snapshot(i)])
    return result
try:
    io.open(f['checkpoint']);assert io.checkpoint_format=='mlx_qwen35_affine_int4_g64_bf16'
    for i,kind in enumerate(c['layer_types']):
        if kind=='linear_attention':
            channels=2*c['linear_num_key_heads']*c['linear_key_head_dim']+c['linear_num_value_heads']*c['linear_value_head_dim']
            states=[[batch,channels,c['linear_conv_kernel_dim']],[batch,c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim']]]
        else:states=[[batch,c['num_key_value_heads'],capacity,c['head_dim']]]*2
        inputs=syms([[batch,width]]+states+[[batch,3],[batch],[batch]],'trained_verify_base_'+str(i)+'_')
        @T.fusion(name='trained_verify_base_'+str(i),boundary='required')
        def forward(hidden,a,b,positions,slots,active):
            if kind=='linear_attention':return Qwen35LinearDecoderLayer(hidden,a,b,active,c,i)
            return Qwen35FullDecoderLayer(hidden,a,b,positions,slots,active,c,i,capacity)
        graph=garnet.compiled_graph();graph.build(forward(*inputs),inputs,f['checkpoint']);base.append(graph)
        inputs=syms([[batch,chunk,width]]+states+[[batch,chunk,3],[batch,chunk],[batch,chunk]],'trained_verify_packed_'+str(i)+'_')
        @T.fusion(name='trained_verify_packed_'+str(i),boundary='required')
        def verify(hidden,a,b,positions,slots,active):
            return PrefillLayer(hidden,a,b,positions,slots,active,c,i,batch,chunk,capacity,kind,
                               parallel_attention=True,parallel_linear=True,attention_capacity_limit=4096)
        graph=garnet.compiled_graph();graph.build_verification_shared(verify(*inputs),inputs,f['checkpoint'],base[-1]);packed.append(graph)
        assert graph.verification_tokens==4
        if i%8==7:print('trained_verification_compiled',i+1,'layers',flush=True)
    for rows in (1,4):
        inputs=syms([[rows,width]],'trained_verify_head_'+str(rows)+'_')
        @T.fusion(name='trained_verify_head_'+str(rows),boundary='required')
        def head(hidden):return norm(hidden,'model.language_model.norm',c)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=0)
        graph=garnet.compiled_graph()
        if heads:graph.build_shared(head(*inputs),inputs,f['checkpoint'],heads[0])
        else:graph.build(head(*inputs),inputs,f['checkpoint'])
        heads.append(graph)
    session.build(base,None);reference.build(base,None)
    for i,graph in enumerate(packed):session.prepare_prefill(graph,i)
    prepared=json.loads(garnet.memory_stats_json())
    assert prepared['metal_allocated_bytes']<=prepared['metal_recommended_working_set_bytes']-256*1024*1024
    print('trained_verification_prepared',json.dumps(prepared),flush=True)
    candidates=[1000,128,220,800]
    hidden=io.stack_tokens([io.embedding([token]) for token in candidates])
    for trial,lengths in enumerate([[0],[1],[2,3],[3],[4]]):
        session.reset();reference.reset()
        # A carried 2+3 case needs five free positions, not just one physical
        # four-token block. Inactive padding may extend beyond the last slot.
        start=capacity-3-max(chunk,sum(lengths)) if capacity==4096 else 0
        for token in (220,1234,912):
            args=[io.embedding([token]),value([float(start)]*3,[1,3]),value([float(start)],[1]),value([1.],[1]),-1]
            session.run(base,*args);reference.run(base,*args);start+=1
        for block,length in enumerate(lengths):
            slots=[float(start+t) for t in range(4)];flags=[1. if t<length else 0. for t in range(4)]
            positions=[s for s in slots for axis in range(3)]
            session.run(packed,hidden,value(positions,[1,4,3]),value(slots,[1,4]),value(flags,[1,4]),-1)
            actual=session.hidden();expected=[];logits=[]
            for t,token in enumerate(candidates):
                reference.run(base,io.embedding([token]),value([slots[t]]*3,[1,3]),value([slots[t]],[1]),value([flags[t]],[1]),-1)
                h=reference.hidden();expected.append(h);logits.append(heads[0].run([h]))
            wanted=io.stack_tokens(expected)
            actual_hash=io.verification_f32_sha256(actual);expected_hash=io.verification_f32_sha256(wanted)
            assert actual_hash==expected_hash,('trained hidden mismatch',capacity,trial,block,length,actual_hash,expected_hash)
            # Snapshot one native tensor at a time; no giant Python float lists.
            state=fingerprint(session);wanted_state=fingerprint(reference)
            assert state==wanted_state,('trained state mismatch',capacity,trial,block,length)
            flat=value(actual.tolist(),[4,width])
            actual_lm=heads[1].run([flat]);wanted_lm=io.stack_tokens(logits)
            lm_hash=io.verification_f32_sha256(actual_lm);wanted_lm_hash=io.verification_f32_sha256(wanted_lm)
            assert lm_hash==wanted_lm_hash,('trained logit mismatch',capacity,trial,block,length,lm_hash,wanted_lm_hash)
            greedy=io.argmax(actual_lm)
            record=dict(trial=trial,block=block,capacity=capacity,physical_tokens=4,active_tokens=length,
                slots=slots,input_ids=candidates,hidden_sha256=actual_hash,reference_hidden_sha256=expected_hash,
                logits_sha256=lm_hash,reference_logits_sha256=wanted_lm_hash,greedy_ids=greedy,
                state_sha256=state,reference_state_sha256=wanted_state)
            records.append(record)
            with open(f['output']+'.cases.jsonl','a') as stream:stream.write(json.dumps(record)+'\n')
            print('trained_verification_case PASS',json.dumps(dict(trial=trial,block=block,active_tokens=length,greedy_ids=greedy)),flush=True)
            start+=length
    result=dict(scope='Trained 27B bounded forward parity only; no draft, acceptance, rollback, coding gate or speed measurement',
        capacity=capacity,batch=1,physical_tokens=4,prepared_memory_snapshot=prepared,backend=garnet.backend_info(),
        checkpoint=f['checkpoint'],cases=records,boundary_history='Three active seed slots with all earlier cache entries zero; not a full 4K prompt' if capacity==4096 else 'Three active seed tokens',passed=True)
    with open(f['output'],'w') as stream:json.dump(result,stream,indent=2)
finally:
    session.close();reference.close()
    for graph in packed+base+heads:graph.close()
    io.close();garnet.shutdown()
    owners=json.loads(garnet.memory_stats_json())
    for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
        assert owners[key]==0,(key,owners[key])
    print('trained_verification_closed',json.dumps(owners),flush=True)
