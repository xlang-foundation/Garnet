"""Learned predefined branched target verifier, bounded head and transaction lifecycle.
Not a speculative generator or speed/process-memory/independent-quality benchmark.
"""
import hashlib,json,os,struct,sys,time,threading
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b.cuda_tree_verify import CUDATreeVerifier

request_path,output_path=map(Path,sys.argv[1:3])
assert not output_path.exists(),'Preserve original learned tree outcome'
request=json.loads(request_path.read_text(encoding='utf-8-sig'))
evidence={'request':request,'events':[],'cases':[],'all_passed':False,
    'performance_or_process_memory_measurement':False,'independent_model_quality_proven':False,
    'branching_or_speculative_generation_accepted':False,'full_goal_met':False}
def save():output_path.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
def event(name):
    evidence['events'].append({'name':name,'unix_seconds':time.time(),'pid':os.getpid()});save()
    print('TREE_VERIFIER_MODEL_EVENT',name,flush=True)
def reject(call):
    try:call()
    except Exception as e:return str(e)
    raise AssertionError('Expected ownership/status rejection')
def canonical(d):
    flat=d.io.storage_view(d.state,[d.batch,d.programs[d.decode_chunk_size].state_width])
    width=d.programs[d.decode_chunk_size].state_width;h=hashlib.sha256()
    for start in range(0,width,131072):
        count=min(131072,width-start)
        value=d.io.slice(flat,start,[d.batch,count]);h.update(d.io.finite_f32_bytes(value))
    return h.hexdigest()
def hidden_words(io,value):
    data=io.finite_f32_bytes(value)
    return data,struct.unpack('<'+str(len(data)//4)+'f',data)

model=verifier=None
try:
    assert request['profile']['batch'] in (1,2,4,8),'Verifier batch profile invalid'
    case=request['case'];assert len(case['images'])==len(case['prompts'])==request['profile']['batch']
    assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
    event('load_begin');model=Model(request['checkpoint'],request['profile'],request['model_id']);d=model.decoder;io=d.io
    event('load_complete');d.reset()
    rows=[model._prepare(case['prompts'][r],case['images'][r]) for r in range(d.batch)]
    assert model.pack_prompt
    hidden=list(io.pack_prompt([r[0] for r in rows],[r[6] for r in rows],[r[7] for r in rows]))
    count=max(len(r[2]) for r in rows)
    positions=[r[1]+[[0,0,0]]*(count-len(r[1])) for r in rows]
    prediction=d.prefill_sequence(hidden,positions,[len(r[2]) for r in rows])
    evidence['real_image_prefix']={'slots':list(d.slots),'first_prediction':list(prediction),'prompt_token_ids':[r[2] for r in rows]}
    before=canonical(d);evidence['canonical_before_sha256']=before;event('real_image_prefill_complete')
    n=request['nodes'];b=d.batch;h=d.config['hidden_size']
    verifier=CUDATreeVerifier(d,n);reference=d.programs[n]
    assert reference.compact_kv_outputs and reference.state_commit is not None and n==d.prefill_chunk_size
    base_state=io.clone(d.state);base_hidden=io.clone(d.last_hidden);base_slots=list(d.slots)
    ref_state=io.clone(d.state);ref_last=io.clone(d.last_hidden)
    tree_input=io.zeros([b,n,h]);tree_active=io.zeros([b,n]);tree_pos=io.zeros([b,n,3]);tree_parent=io.zeros([b,n]);tree_prefix=io.zeros([b,1])
    ref_input=io.zeros([b,n,h]);ref_active=io.zeros([b,n]);ref_pos=io.zeros([b,n,3]);ref_slots=io.zeros([b,n]);ref_hidden=io.zeros([b,n,h])
    chosen_mask=io.zeros([b,n]);chosen_hidden=io.clone(base_hidden);chosen_norm=io.zeros([b,h])
    ids=[[int(prediction[r]) if t==0 else 100+(t*37+r*19)%10000 for r in range(b)] for t in range(n)]
    io.embedding_chunk(ids,tree_input);io.update(tree_prefix,[float(v) for v in base_slots])
    def restore():
        assert not verifier.pending
        io.copy_into(base_state,d.state);io.copy_into(base_hidden,d.last_hidden);d.slots=list(base_slots);d.last_packed=None;d.poisoned=False
    def equal_state(a,e):
        count=0;ha=hashlib.sha256();he=hashlib.sha256()
        for start in range(0,verifier.program.state_width,131072):
            width=min(131072,verifier.program.state_width-start)
            actual=io.finite_f32_bytes(io.slice(a,start,[b,width]));expected=io.finite_f32_bytes(io.slice(e,start,[b,width]))
            if actual!=expected:
                aa=output_path.with_name(output_path.stem+'-actual-state-chunk.f32');ee=output_path.with_name(output_path.stem+'-reference-state-chunk.f32')
                aa.write_bytes(actual);ee.write_bytes(expected)
                evidence['state_difference']={'start_per_row':start,'width_per_row':width,'actual':str(aa),'reference':str(ee)};save()
                raise AssertionError('Full branched canonical state differs from original serial target')
            ha.update(actual);he.update(expected);count+=b*width
        return {'exact_canonical_words':count,'actual_sha256':ha.hexdigest(),'reference_sha256':he.hexdigest()}
    base_hidden_bytes=io.finite_f32_bytes(base_hidden)
    for case_index,branch in enumerate(request['tree_cases']):
        restore();parents=branch['parents'];active=branch['active'];chosen=branch['selected_last_nodes']
        node_positions=[];paths=[];counts=[];last_active=[]
        for r in range(b):
            per_node=[]
            for node in range(n):
                path=[];current=node
                while current>=0:path.append(current);current=int(parents[r*n+current])
                path.reverse();depth=sum(int(active[r*n+j]) for j in path)
                per_node.append([rows[r][3]+depth-1]*3 if active[r*n+node] else [0,0,0])
            node_positions.append(per_node);path=[];current=chosen[r]
            while current>=0:path.append(current);current=int(parents[r*n+current])
            path.reverse();paths.append(path);used=[j for j in path if active[r*n+j]];counts.append(len(used));last_active.append(used[-1] if used else -1)
        io.update(tree_active,active);io.update(tree_parent,parents);io.update(tree_pos,[v for row in node_positions for p in row for v in p])
        if case_index==0:first_positions=[v for row in node_positions for p in row for v in p]
        verifier.verify(tree_input,tree_active,tree_pos,tree_parent,tree_prefix)
        assert canonical(d)==before and io.finite_f32_bytes(d.last_hidden)==base_hidden_bytes and d.slots==base_slots
        actual_predictions=list(verifier.predictions());assert len(actual_predictions)==b*n and all(0<=v<d.config['vocab_size'] for v in actual_predictions)
        for r in range(b):
            for node in range(n):
                if not active[r*n+node]:assert actual_predictions[r*n+node]==0,'Inactive zero-hidden head tie must choose lowest token0'
        ref_ids=[[0 for r in range(b)] for t in range(n)];rf=[];rp=[];rs=[]
        for r in range(b):
            slot=base_slots[r]
            for t in range(n):
                flag=0;position=[0,0,0]
                if t<len(paths[r]):
                    node=paths[r][t];ref_ids[t][r]=ids[node][r];flag=int(active[r*n+node]);position=node_positions[r][node]
                rf.append(float(flag));rp.extend(position);rs.append(float(slot));slot+=flag
        io.embedding_chunk(ref_ids,ref_input);io.update(ref_active,rf);io.update(ref_pos,rp);io.update(ref_slots,rs)
        io.copy_into(base_state,ref_state);io.copy_into(base_hidden,ref_last)
        original=reference.run([ref_input,d.state,ref_active,ref_pos,ref_slots]);io.slice_into(original,0,ref_hidden)
        reference.commit_all(original,ref_state,ref_slots,ref_active);io.copy_last_chunk(ref_hidden,ref_active,ref_last)
        expected_hidden=io.finite_f32_bytes(ref_last)
        io.copy_into(base_hidden,chosen_hidden);mask=[0.]*(b*n)
        for r in range(b):
            if last_active[r]>=0:mask[r*n+last_active[r]]=1.
        io.update(chosen_mask,mask);io.copy_last_chunk(verifier.program.hidden,chosen_mask,chosen_hidden)
        actual_hidden=io.finite_f32_bytes(chosen_hidden)
        if actual_hidden!=expected_hidden:
            a=output_path.with_name(output_path.stem+'-actual-selected-hidden.f32');e=output_path.with_name(output_path.stem+'-reference-selected-hidden.f32')
            a.write_bytes(actual_hidden);e.write_bytes(expected_hidden);evidence['selected_hidden_difference']={'actual':str(a),'reference':str(e)};save()
            raise AssertionError('Selected branched hidden differs from original serial target before token head')
        normalized_reference=d.head.run_device([ref_last]);expected_predictions=list(io.predict(normalized_reference))
        evidence['pending_prediction_check']={'case':case_index,'selected_last_active_nodes':last_active,'tree':[actual_predictions[r*n+last_active[r]] if last_active[r]>=0 else None for r in range(b)],'serial':expected_predictions};save()
        for r in range(b):
            if last_active[r]>=0 and actual_predictions[r*n+last_active[r]]!=expected_predictions[r]:
                io.zero(chosen_norm);io.copy_last_chunk(verifier.program.head_input,chosen_mask,chosen_norm)
                a=output_path.with_name(output_path.stem+'-actual-head-input.f32');e=output_path.with_name(output_path.stem+'-reference-head-input.f32')
                a.write_bytes(io.finite_f32_bytes(chosen_norm));e.write_bytes(io.finite_f32_bytes(normalized_reference))
                evidence['head_input_difference_snapshots']={'actual':str(a),'reference':str(e),'row':r};save()
                raise AssertionError('Branched tree head token differs from original serial target head')
        if case_index==0:
            lifecycle={'parent_reset_pending':reject(d.reset),'parent_close_pending':reject(d.close),
                'ordinary_chunk_pending':reject(lambda:d.chunk_hidden([],[],[],False)),
                'program_close_pending':reject(verifier.program.close),'wrong_owner_prediction':reject(lambda:verifier.program.predictions(object()))}
            errors=[]
            def foreign():
                try:verifier.rollback()
                except Exception as error:errors.append(str(error))
            thread=threading.Thread(target=foreign);thread.start();thread.join(10)
            assert not thread.is_alive() and len(errors)==1 and verifier.pending and d.state_transaction is verifier
            lifecycle['foreign_thread_rollback']=errors[0]
            lifecycle['invalid_host_selection']=reject(lambda:verifier.commit([n]*b))
            assert verifier.pending and not d.poisoned and canonical(d)==before
            verifier.rollback();assert d.state_transaction is None and not d.poisoned and canonical(d)==before and io.finite_f32_bytes(d.last_hidden)==base_hidden_bytes and d.slots==base_slots
            lifecycle['stale_predictions_after_rollback']=reject(verifier.predictions)
            # Legal buffer prefix that disagrees with the canonical clock must
            # never expose predictions or cause a state write.
            io.update(tree_prefix,[float(v+1) for v in base_slots])
            lifecycle['stale_legal_prefix']=reject(lambda:verifier.verify(tree_input,tree_active,tree_pos,tree_parent,tree_prefix))
            assert not verifier.pending and not d.poisoned and canonical(d)==before
            io.update(tree_prefix,[float(v) for v in base_slots]);verifier.verify(tree_input,tree_active,tree_pos,tree_parent,tree_prefix)
            corrupt_active=list(active);corrupt_active[(b-1)*n]=.5;io.update(tree_active,corrupt_active)
            lifecycle['native_known_no_write_abort']=reject(lambda:verifier.commit(chosen))
            assert not verifier.pending and not d.poisoned and canonical(d)==before and io.finite_f32_bytes(d.last_hidden)==base_hidden_bytes and d.slots==base_slots
            io.update(tree_active,active);verifier.verify(tree_input,tree_active,tree_pos,tree_parent,tree_prefix)
            evidence['lifecycle']=lifecycle
        result=verifier.commit(chosen);comparison=equal_state(d.state,ref_state)
        assert result['accepted_input_counts']==counts and d.slots==[v+c for v,c in zip(base_slots,counts)]
        assert io.finite_f32_bytes(d.last_hidden)==expected_hidden and not d.poisoned and not verifier.pending and d.state_transaction is None
        comparison.update({'case':case_index,'accepted_input_counts':counts,'last_active_nodes':last_active,'commit':result,
            'selected_prediction_ids':[actual_predictions[r*n+last_active[r]] if last_active[r]>=0 else None for r in range(b)],'serial_prediction_ids':expected_predictions,
            'all_node_prediction_ids':actual_predictions,'predicted_rows':b*n,'inactive_zero_tie_checks':sum(v==0 for v in active),
            'selected_hidden_before_head_exact':True,'last_hidden_exact':True,'clock_exact':True})
        evidence['cases'].append(comparison);save();event('branched_case_'+str(case_index)+'_passed')
    restore();evidence['memory_scoped_diagnostic']=verifier.memory_info();evidence['correctness_only_reference_canonical_copy_bytes']=2*io.tensor_bytes(base_state)
    first=request['tree_cases'][0];io.update(tree_parent,first['parents']);io.update(tree_active,first['active']);io.update(tree_pos,first_positions)
    # Force an exception AFTER real native state writes, then require poison,
    # transaction retirement, rejection of further work, and explicit reset.
    verifier.verify(tree_input,tree_active,tree_pos,tree_parent,tree_prefix)
    real=verifier.helper
    class CompletionFault:
        def commit(self,*args):
            code=real.commit(*args)
            if code==0:raise RuntimeError('INJECTED_AFTER_REAL_NATIVE_CANONICAL_WRITE')
            return code
        def close(self):return real.close()
    verifier.helper=CompletionFault()
    reason=reject(lambda:verifier.commit(first['selected_last_nodes']))
    assert 'INJECTED_AFTER_REAL_NATIVE_CANONICAL_WRITE' in reason and d.poisoned and not verifier.pending and d.state_transaction is None
    assert canonical(d)!=before,'Fault injection must follow actual canonical writes'
    evidence['unknown_completion_poison']={'reason':reason,'ordinary_chunk_rejection':reject(lambda:d.chunk_hidden([],[],[],False)),
        'new_tree_rejection':reject(lambda:verifier.verify(tree_input,tree_active,tree_pos,tree_parent,tree_prefix))}
    d.reset();assert not d.poisoned and d.slots==[0]*b
    for start in range(0,verifier.program.state_width,131072):
        width=min(131072,verifier.program.state_width-start);raw=io.finite_f32_bytes(io.slice(d.state,start,[b,width]));assert raw==b'\x00'*len(raw)
    assert io.finite_f32_bytes(d.last_hidden)==b'\x00'*(4*b*h)
    evidence['explicit_reset_after_unknown_completion_zeroes_state_hidden_clock']=True
    evidence['all_passed']=True
    evidence['scope']='Learned real-image prefixes and predefined dynamic branched trees, matching serial P64 full-state/last_hidden/clock and selected greedy head IDs; successful bounded head to512 rows and inactive tie0, rollback/ownership/thread/host-selection/stale-prefix guards, injected AFTER-WRITE failure poison and reset. Test-only canonical reference copies. No learned MTP proposals/adaptive acceptance/generator, broad independent quality, speed or process-memory acceptance.'
    save();event('branched_tree_verifier_passed')
except Exception as error:
    evidence['failure']=str(error);save();raise
finally:
    if verifier is not None:verifier.close()
    if model is not None:model.close()
    event('closed')
