"""Full hybrid forward equality: explicit FP32 verification vs ordinary steps."""
import hashlib,json,math,os,struct,sys
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet,tensor
from xModel.qwen35.vl_0_8b.decoder_layer import T,norm,Qwen35LinearDecoderLayer,Qwen35FullDecoderLayer
from xModel.qwen35.vl_0_8b.prefill_layer import PrefillLayer
with open(sys.argv[1]) as stream:f=json.load(stream)
c=f['config'];width=c['hidden_size'];checks=0;words=0
assert 'verification_attention_fp32' in garnet.backend_info()
assert 'compiled_verification_fp32' in garnet.backend_info()
io=garnet.tensor_io()

def value(data,shape):return tensor.tensor(data,shape=shape,dtype=tensor.float32)
def syms(shapes,name):return [tensor.input(name+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
def flatten(data):
    out=[]
    for x in data:
        if isinstance(x,list):out.extend(flatten(x))
        else:out.append(x)
    return out
def equal(a,b,label):
    global checks,words
    a=flatten(a);b=flatten(b)
    assert len(a)==len(b),label+' size'
    for i in range(len(a)):
        if struct.pack('<f',a[i])!=struct.pack('<f',b[i]):
            raise AssertionError(label+' word '+str(i)+' '+str(a[i])+' != '+str(b[i]))
    checks+=1;words+=len(a)
def snapshot(owner):return [[part.tolist() for part in owner.snapshot(i)] for i in range(4)]
def reject(action,label):
    global checks
    failed=False
    try:action()
    except Exception:failed=True
    assert failed,'Missing rejection: '+label
    checks+=1

probe=[0.,-0.,1.,-2.5,.125]
assert io.verification_f32_sha256(value(probe,[5]))==hashlib.sha256(struct.pack('<5f',*probe)).hexdigest()

for batch,chunk,capacity in [(1,1,16),(1,2,16),(1,3,16),(1,4,16),(2,1,16),(2,2,16),(4,1,16),(1,4,4096)]:
    base=[];packed=[];heads=[];extra=[];session=garnet.decoder_session();reference=garnet.decoder_session();other=garnet.decoder_session()
    label='B'+str(batch)+'T'+str(chunk)+'C'+str(capacity)
    states=[[[batch,320,4],[batch,4,128,16]],[[batch,2,capacity,32],[batch,2,capacity,32]]]
    try:
        for i in range(4):
            shapes=[[batch,width]]+states[i%2]+[[batch,3],[batch],[batch]]
            inputs=syms(shapes,'verify_base_'+label+'_'+str(i)+'_')
            @T.fusion(name='verify_base_'+label+'_'+str(i),boundary='required')
            def forward(hidden,a,b,positions,slots,active):
                if i%2==0:return Qwen35LinearDecoderLayer(hidden,a,b,active,c,i)
                return Qwen35FullDecoderLayer(hidden,a,b,positions,slots,active,c,i,capacity)
            graph=garnet.compiled_graph();graph.build(forward(*inputs),inputs,f['checkpoint']);base.append(graph)
            assert graph.verification_tokens==0
            shapes=[[batch,chunk,width]]+states[i%2]+[[batch,chunk,3],[batch,chunk],[batch,chunk]]
            inputs=syms(shapes,'verify_packed_'+label+'_'+str(i)+'_')
            @T.fusion(name='verify_packed_'+label+'_'+str(i),boundary='required')
            def verify(hidden,a,b,positions,slots,active):
                return PrefillLayer(hidden,a,b,positions,slots,active,c,i,batch,chunk,capacity,
                    'linear_attention' if i%2==0 else 'full_attention',parallel_attention=True,parallel_linear=True,
                    attention_capacity_limit=4096)
            graph=garnet.compiled_graph();graph.build_verification_shared(verify(*inputs),inputs,f['checkpoint'],base[-1]);packed.append(graph)
            assert graph.verification_tokens==chunk
        # Separate 2D heads exercise every candidate position, not just argmax.
        for rows in (batch,batch*chunk):
            inputs=syms([[rows,width]],'verify_head_'+label+'_'+str(rows)+'_')
            @T.fusion(name='verify_head_'+label+'_'+str(rows),boundary='required')
            def head(hidden):return norm(hidden,'model.language_model.norm',c)*T.unary_op('qwen_text_head',weight_name='lm_head.weight',bf16_weight=0)
            graph=garnet.compiled_graph()
            if heads:graph.build_shared(head(*inputs),inputs,f['checkpoint'],heads[0])
            else:graph.build(head(*inputs),inputs,f['checkpoint'])
            heads.append(graph)
        session.build(base,None);reference.build(base,None);other.build(base,None)
        for i,graph in enumerate(packed):session.prepare_prefill(graph,i)
        # Zero through full active prefixes and different row lengths. Near C4096
        # includes the final legal slot; other profiles carry state across blocks.
        for trial in range(chunk+2):
            session.reset();reference.reset()
            starts=[float(capacity-chunk) if capacity==4096 else 1.0]*batch
            seed=[math.sin((i+3)*.19)*.2 for i in range(batch*width)]
            seed_slots=[s-1 for s in starts]
            args=[value(seed,[batch,width]),value([s for s in seed_slots for axis in range(3)],[batch,3]),
                  value(seed_slots,[batch]),value([1.]*batch,[batch]),-1]
            session.run(base,*args);reference.run(base,*args)
            for block in range(1 if capacity==4096 else 2):
                lengths=[(trial+row)%(chunk+1) for row in range(batch)]
                if trial==chunk+1:lengths=[chunk]*batch
                flags=[1.0 if t<lengths[row] else 0.0 for row in range(batch) for t in range(chunk)]
                slots=[starts[row]+t for row in range(batch) for t in range(chunk)]
                positions=[s for s in slots for axis in range(3)]
                hidden=[math.sin((i+trial*29+block*31)*.137)*.2 if flags[i//width] else float('nan') for i in range(batch*chunk*width)]
                session.run(packed,value(hidden,[batch,chunk,width]),value(positions,[batch,chunk,3]),value(slots,[batch,chunk]),value(flags,[batch,chunk]),-1)
                actual=flatten(session.hidden().tolist());expected=[0.]*(batch*chunk*width);logits=[0.]*(batch*chunk*128)
                for t in range(chunk):
                    h=[hidden[(row*chunk+t)*width+k] for row in range(batch) for k in range(width)]
                    sl=[slots[row*chunk+t] for row in range(batch)]
                    ac=[flags[row*chunk+t] for row in range(batch)]
                    reference.run(base,value(h,[batch,width]),value([s for s in sl for axis in range(3)],[batch,3]),value(sl,[batch]),value(ac,[batch]),-1)
                    result=flatten(reference.hidden().tolist())
                    lm=flatten(heads[0].run([value(result,[batch,width])]).tolist())
                    for row in range(batch):
                        expected[(row*chunk+t)*width:(row*chunk+t+1)*width]=result[row*width:(row+1)*width]
                        logits[(row*chunk+t)*128:(row*chunk+t+1)*128]=lm[row*128:(row+1)*128]
                equal(actual,expected,label+' all hidden')
                equal(heads[1].run([value(actual,[batch*chunk,width])]).tolist(),logits,label+' all logits')
                equal(snapshot(session),snapshot(reference),label+' all states')
                starts=[starts[row]+lengths[row] for row in range(batch)]
        # Invalid control metadata must leave every native state unchanged.
        before=snapshot(session);untouched=snapshot(other)
        hidden=value([.1]*(batch*chunk*width),[batch,chunk,width])
        positions=value([0.]*(batch*chunk*3),[batch,chunk,3])
        good_slots=[float(t) for row in range(batch) for t in range(chunk)];good_flags=[1.]*(batch*chunk)
        mutations=[]
        bad=list(good_flags);bad[0]=.5;mutations.append((good_slots,bad))
        bad=list(good_slots);bad[0]=-1.;mutations.append((bad,good_flags))
        bad=list(good_slots);bad[0]=float(capacity);mutations.append((bad,good_flags))
        if chunk>1:
            bad=list(good_slots);bad[1]=bad[0];mutations.append((bad,good_flags))
            bad=list(good_slots);bad[1]=bad[0]+2.;mutations.append((bad,good_flags))
            bad=list(good_flags);bad[0]=0.;mutations.append((good_slots,bad))
        for sl,ac in mutations:
            reject(lambda:session.run(packed,hidden,positions,value(sl,[batch,chunk]),value(ac,[batch,chunk]),-1),label+' controls')
            equal(snapshot(session),before,label+' reject state unchanged')
            equal(snapshot(other),untouched,label+' private other unchanged')
        valid_args=[hidden,positions,value(good_slots,[batch,chunk]),value(good_flags,[batch,chunk]),-1]
        bad_positions=[0.]*(batch*chunk*3);bad_positions[0]=float('nan')
        reject(lambda:session.run(packed,hidden,value(bad_positions,[batch,chunk,3]),valid_args[2],valid_args[3],-1),label+' nonfinite position')
        equal(snapshot(session),before,label+' position reject unchanged')
        os.environ['GARNET_METAL_PROFILE_OPS']='1'
        try:reject(lambda:session.run(packed,*valid_args),label+' operation splitting')
        finally:os.environ['GARNET_METAL_PROFILE_OPS']='0'
        equal(snapshot(session),before,label+' profiling reject unchanged')
        wrong=list(packed);wrong[0],wrong[2]=wrong[2],wrong[0]
        reject(lambda:session.run(wrong,*valid_args),label+' different layer weights')
        equal(snapshot(session),before,label+' family reject unchanged')
        if batch==1 and chunk==4 and capacity==16:
            for length in (4,5):
                shapes=[[batch,length,width]]+states[0]+[[batch,length,3],[batch,length],[batch,length]]
                inputs=syms(shapes,'verify_contract_'+str(length)+'_')
                @T.fusion(name='verify_contract_'+str(length),boundary='required')
                def contract(hidden,a,b,positions,slots,active):
                    return PrefillLayer(hidden,a,b,positions,slots,active,c,0,batch,length,capacity,'linear_attention',
                        parallel_attention=True,parallel_linear=length==5)
                graph=garnet.compiled_graph();extra.append(graph)
                reject(lambda:graph.build_verification_shared(contract(*inputs),inputs,f['checkpoint'],base[0]),label+' unrolled/five tokens')
                assert not graph.ready and graph.verification_tokens==0
            inputs=syms([[batch,chunk,width]]+states[0]+[[batch,chunk,3],[batch,chunk],[batch,chunk]],'verify_mixed_')
            @T.fusion(name='verify_mixed',boundary='required')
            def mixed(hidden,a,b,positions,slots,active):
                return PrefillLayer(hidden,a,b,positions,slots,active,c,0,batch,chunk,capacity,'linear_attention',parallel_linear=True)
            graph=garnet.compiled_graph();extra.append(graph)
            graph.build_shared(mixed(*inputs),inputs,f['checkpoint'],base[0])
            reject(lambda:session.run([graph]+packed[1:],*valid_args),label+' mixed projection modes')
            equal(snapshot(session),before,label+' mixed modes unchanged')
        if batch==2 and chunk==2:
            inputs=syms([[2,4,width]]+states[0]+[[2,4,3],[2,4],[2,4]],'verify_eight_rows_')
            @T.fusion(name='verify_eight_rows',boundary='required')
            def eight(hidden,a,b,positions,slots,active):
                return PrefillLayer(hidden,a,b,positions,slots,active,c,0,2,4,capacity,'linear_attention',parallel_linear=True)
            graph=garnet.compiled_graph();extra.append(graph)
            reject(lambda:graph.build_verification_shared(eight(*inputs),inputs,f['checkpoint'],base[0]),label+' eight physical rows')
        # Verify retained shared weights also survive closure of their source
        # for standalone use. A session tied to that closed owner must reject.
        zero1=io.zeros(states[0][0]);zero2=io.zeros(states[0][1])
        standalone_args=[hidden,zero1,zero2,positions,valid_args[2],valid_args[3]]
        saved=packed[0].run(standalone_args).tolist()
        base[0].close()
        reject(lambda:session.run(packed,*valid_args),label+' closed source')
        equal(snapshot(session),before,label+' closed source reject unchanged')
        equal(packed[0].run(standalone_args).tolist(),saved,label+' retained standalone weights')
        inputs=syms([[batch,width]]+states[0]+[[batch,3],[batch],[batch]],'verify_rebuild_'+label+'_')
        @T.fusion(name='verify_rebuild_'+label,boundary='required')
        def rebuilt(hidden,a,b,positions,slots,active):return Qwen35LinearDecoderLayer(hidden,a,b,active,c,0)
        base[0].build(rebuilt(*inputs),inputs,f['checkpoint'])
        reject(lambda:session.run(packed,*valid_args),label+' rebuilt source')
        equal(snapshot(session),before,label+' rebuilt source reject unchanged')
        print('compiled_verification_profile PASS',label,'checks',checks,'words',words,flush=True)
    finally:
        session.close();reference.close();other.close()
        for graph in packed+base+heads+extra:graph.close()
        assert all(graph.verification_tokens==0 for graph in packed+base+heads+extra)
print('compiled_verification PASS',json.dumps(dict(checks=checks,fp32_words=words,model_speed_claim=False)),flush=True)
io.close()
garnet.shutdown()
owners=json.loads(garnet.memory_stats_json())
for key in ('garnet_live_matrix_owners','garnet_live_affine_weight_owners','garnet_live_bf16_weight_owners','garnet_live_pipeline_sets'):
    assert owners[key]==0,(key,owners[key])
print('compiled_verification_owners PASS',json.dumps(owners),flush=True)
