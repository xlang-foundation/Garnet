"""Matching serialP64 versus real N64 beam8 batched-indexed tree generation.

Correctness/lifecycle only. Hashes/downloads/fault injection invalidate timing
and process-memory comparison. Original feature-exact MTP gate stays failed.
"""
import array,hashlib,json,os,sys,threading,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from xModel.qwen35.vl_0_8b.cuda_tree_speculative import CUDATreeSpeculativeSession
from qwen35_tree_parent_reservation import reserve_tree_parent

request_path,output_path=map(Path,sys.argv[1:3]);assert not output_path.exists(),'Preserve first generation evidence'
request=json.loads(request_path.read_text(encoding='utf-8-sig'))
assert request['fixed_rms'] and os.environ.get('GARNET_QWEN35_FIXED_RMS')=='1','Fixed RMS environment missing'
evidence={'request':request,'events':[],'ordinary':[],'tree':[],'failed_quality_gates':[],'all_passed':False,
    'original_mtp_exact_feature_gate_passed':False,'performance_or_process_memory_measurement':False,'full_goal_met':False}
def save():output_path.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
def event(name):
    evidence['events'].append({'name':name,'pid':os.getpid(),'unix_seconds':time.time()});save();print('TREE_GENERATION_EVENT',name,flush=True)
def signature(result):return {k:result[k] for k in ('prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','visual_tokens_per_row')}
def reject(call):
    try:call()
    except Exception as error:return str(error)
    raise AssertionError('Expected tree generation ownership/poison rejection')
def digest_tensor(io,value,b,width):
    flat=io.storage_view(value,[b,width]);out=hashlib.sha256()
    for start in range(0,width,32768):out.update(io.finite_f32_bytes(io.slice(flat,start,[b,min(32768,width-start)])))
    return out.hexdigest()
def state(model,file=None):
    d=model.decoder;b=d.batch;io=d.io;width=d.programs[d.decode_chunk_size].state_width;digest=hashlib.sha256()
    stream=file.open('xb') if file is not None else None
    try:
        for start in range(0,width,32768):
            raw=io.finite_f32_bytes(io.slice(d.state,start,[b,min(32768,width-start)]));digest.update(raw)
            if stream is not None:stream.write(raw)
    finally:
        if stream is not None:stream.close()
    hidden=io.finite_f32_bytes(d.last_hidden)
    return {'canonical_sha256':digest.hexdigest(),'canonical_words':b*width,'last_hidden_sha256':hashlib.sha256(hidden).hexdigest(),'slots':list(d.slots)}
def compare_full_state(model,reference_file):
    # Retain complete ACTUAL state for offline NumPy analysis after the GPU
    # worker closes. Bytewise exactness is the identical original strict gate.
    d=model.decoder;b=d.batch;io=d.io;width=d.programs[d.decode_chunk_size].state_width
    file=output_path.with_name(output_path.stem+'-actual32-canonical.f32');different_chunks=0
    with reference_file.open('rb') as source,file.open('xb') as sink:
        for start in range(0,width,32768):
            count=min(32768,width-start);actual=io.finite_f32_bytes(io.slice(d.state,start,[b,count]));expected=source.read(len(actual));assert len(expected)==len(actual)
            sink.write(actual)
            if actual!=expected:different_chunks+=1
        assert not source.read(1)
    output_path.with_name(output_path.stem+'-actual32-last-hidden.f32').write_bytes(io.finite_f32_bytes(d.last_hidden))
    return {'exact':different_chunks==0,'different_chunks':different_chunks,'actual_full_state':str(file),
        'offline_word_analysis_pending':True,'state_width_per_row':width,'batch':b,'capacity':d.capacity,
        'layout':d.programs[d.decode_chunk_size].layout,'prompt_lengths':request['case'].get('prompt_lengths_from_result'),
        'next_slots':list(d.slots),'scope':'Identical bytewise full-state exact gate; word/relative/layer/position analysis deferred until model worker closed.'}
def row_major_bytes(io,value,batch,width):
    flat=io.storage_view(value,[batch,width]);out=bytearray(batch*width*4)
    for start in range(0,width,32768):
        count=min(32768,width-start);chunk=io.finite_f32_bytes(io.slice(flat,start,[batch,count]))
        for row in range(batch):
            dest=(row*width+start)*4;source=row*count*4
            out[dest:dest+count*4]=chunk[source:source+count*4]
    return bytes(out)

def fresh_prefill(model,case):
    s=model.speculative;d=model.decoder;io=d.io;s.reset()
    rows=[model._prepare(case['prompts'][r],case['images'][r]) for r in range(d.batch)]
    hidden=list(io.pack_prompt([r[0] for r in rows],[r[6] for r in rows],[r[7] for r in rows]))
    length=max(len(r[2]) for r in rows);positions=[r[1]+[[0,0,0]]*(length-len(r[1])) for r in rows]
    return s.prefill(hidden,positions,[len(r[2]) for r in rows],[r[3] for r in rows])
model=None;saved_factory=specmod.CUDASpeculativeSession
try:
    historical=json.loads(Path(request['historical_reference']).read_text(encoding='utf-8-sig'))
    case=request['case'];assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
    reference_file=output_path.with_name(output_path.stem+'-ordinary32-canonical.f32')
    if request.get('ordinary_reference'):
        prior=json.loads(Path(request['ordinary_reference']).read_text(encoding='utf-8-sig'))
        assert prior['events'][-1]['name']=='closed' and len(prior['ordinary'])==4
        assert prior['request']['ordinary_profile']==request['ordinary_profile'] and prior['request']['case']==case
        assert prior['request']['checkpoint']==request['checkpoint'] and prior['request']['fixed_rms']==request['fixed_rms']
        evidence['ordinary']=prior['ordinary'];evidence['serial_projection_rows_used']=prior['serial_projection_rows_used']
        evidence['serial_reference_mode']=prior['serial_reference_mode']
        evidence['ordinary_reference_reused_from']=request['ordinary_reference']
        evidence['ordinary_fresh_in_this_worker']=False
        reference_file=Path(request['ordinary_reference_file'])
        event('verified_closed_ordinary_reference_reused')
    else:
        evidence['ordinary_fresh_in_this_worker']=True
        event('ordinary_load_begin');model=Model(request['checkpoint'],request['ordinary_profile'],request['model_id']);event('ordinary_load_complete')
        d=model.decoder;original_step=d.step
        assert d.prefill_chunk_size==64 and d.decode_chunk_size==8 and not d.prefill_mode
        serial_steps=[]
        def p64_step(*args,**kwargs):
            assert not d.prefill_mode
            d.prefill_mode=True
            try:
                assert d.chunk_size==64
                value=original_step(*args,**kwargs);serial_steps.append(64);return value
            finally:d.prefill_mode=False
        d.step=p64_step
        evidence['serial_reference_mode']='Existing P64 program, one active decode input per step; explicit test-only override, not the fast baseline.'

        for limit in request['output_limits']:
            result=model.infer_batch(case['prompts'],case['images'],limit)
            observed=state(model,reference_file if limit==32 else None)
            if limit==32:output_path.with_name(output_path.stem+'-ordinary32-last-hidden.f32').write_bytes(model.decoder.io.finite_f32_bytes(model.decoder.last_hidden))
            assert observed['slots']==[p+s-1 for p,s in zip(result['prompt_tokens'],result['sampled_tokens_including_stop'])]
            previous=next(x for x in historical['ordinary'] if x['limit']==limit)
            historical_match=signature(result)==signature(previous['result'])
            if not historical_match:evidence['failed_quality_gates'].append({'limit':limit,'gate':'exact historical default emitted/prompt/stop outputs under fixed RMS'})
            evidence['ordinary'].append({'limit':limit,'result':result,'state':observed,'historical_output_signature_exact':historical_match,
                'historical_complete_state_exact':observed==previous['state']});save()
        evidence['serial_projection_rows_used']=serial_steps;assert serial_steps and set(serial_steps)=={64}
        model.close();model=None;event('ordinary_closed')
    def factory(*args):return CUDATreeSpeculativeSession(*args,nodes=request['nodes'],beam=request['beam'],share_parent_storage=request['share_parent_storage'],batch_frontier_heads=request['batch_frontier_heads'],batch_indexed_heads=True,scored_indexed_heads=request['scored_indexed_heads'])
    specmod.CUDASpeculativeSession=factory;event('tree_load_begin')
    try:
        with reserve_tree_parent(request):model=Model(request['checkpoint'],request['tree_profile'],request['model_id'])
    finally:specmod.CUDASpeculativeSession=saved_factory
    event('tree_load_complete');s=model.speculative;d=model.decoder;io=d.io
    memory=model.memory_info();parent=memory['decoder'];target=memory['speculative']['target_verifier']['tree']
    assert parent['shared_output_pool_bytes']>=d.batch*target['output_width']*4
    assert parent['shared_activation_pool_bytes']>=target['graph']['context_workspace_required_bytes']
    assert target['graph']['owned_device_output_bytes']==0 and target['graph']['external_workspace_bound']
    assert s.scored_indexed_heads and s.nodes==16 and s.beam==2
    evidence['fixed_node_score_policy']='anchored_binary_top2_conditional_nll'
    evidence['reference_native_epoch']=request['ordinary_reference_native_sha256']
    evidence['current_native_epoch']=request['current_native_sha256']
    for kw in ({'batch_indexed_heads':False,'scored_indexed_heads':True},
               {'batch_indexed_heads':True,'scored_indexed_heads':1},
               {'batch_indexed_heads':True,'scored_indexed_heads':True,'beam':1}):
        args={'nodes':16,'beam':2,'share_parent_storage':True};args.update(kw)
        reject(lambda:CUDATreeSpeculativeSession(d,request['checkpoint'],s.stops,7,8192,True,**args))
    evidence['invalid_score_policy_constructor_before_allocation']=True
    for limit in request['output_limits']:
        result=model.infer_batch(case['prompts'],case['images'],limit);observed=state(model)
        expected=next(x for x in evidence['ordinary'] if x['limit']==limit)
        match=signature(result)==signature(expected['result']);canonical_exact=observed==expected['state']
        row={'limit':limit,'result':result,'state':observed,'signature_exact':match,'canonical_hidden_clock_exact':canonical_exact}
        if limit==32:row['full_canonical_comparison']=compare_full_state(model,reference_file)
        evidence['tree'].append(row)
        if not match:evidence['failed_quality_gates'].append({'limit':limit,'gate':'exact original ordinaryD4 emitted/prompt/stop outputs'})
        if not canonical_exact:evidence['failed_quality_gates'].append({'limit':limit,'gate':'exact original ordinaryD4 complete canonical/last_hidden/clock'})
        assert not d.poisoned and not s.poisoned and not s.verifier.pending and d.state_transaction is None and d.slots==s.slots
        assert d.slots==[p+count-1 for p,count in zip(result['prompt_tokens'],result['sampled_tokens_including_stop'])]
        save();print('TREE_GENERATION_REAL_CASE_RECORDED',limit,match,canonical_exact,flush=True)
    repeated=model.infer_batch(case['prompts'],case['images'],32);repeat_state=state(model)
    prior=next(x for x in evidence['tree'] if x['limit']==32)
    assert signature(repeated)==signature(prior['result']) and repeat_state==prior['state']
    assert repeated['speculative_metrics']==prior['result']['speculative_metrics']
    evidence['exact_tree_replay']=True

    # Raise only AFTER real initial B*beam candidate expansion, after the
    # inherited prefill has marked ready. Both owners must still be poisoned.
    original_io=s.io;expanded=[]
    class ExpansionFault:
        def __getattr__(self,name):return getattr(original_io,name)
        def head_candidates(self,rows,count,token_cache):
            result=original_io.head_candidates(rows,count,token_cache)
            if not token_cache and len(rows)==s.batch*s.beam:
                expanded.append(list(rows));raise RuntimeError('INJECTED_AFTER_REAL_INITIAL_SHORTLIST_EXPANSION')
            return result
    s.io=ExpansionFault()
    try:prefill_failure=reject(lambda:fresh_prefill(model,case))
    finally:s.io=original_io
    assert expanded and 'INJECTED_AFTER_REAL_INITIAL_SHORTLIST_EXPANSION' in prefill_failure
    assert d.poisoned and s.poisoned and d.state_transaction is None and not s.verifier.pending
    evidence['after_real_initial_expansion_poison']={'reason':prefill_failure,'expanded_source_rows':expanded,'generate_rejected':reject(lambda:s.generate(8))}
    s.reset();assert not d.poisoned and not s.poisoned and d.slots==s.slots==[0]*d.batch
    for value,width in ((d.state,d.programs[d.decode_chunk_size].state_width),(d.last_hidden,s.hidden_size),(s.key,s.cache_width),(s.value,s.cache_width),(s.last_hidden,s.hidden_size)):
        flat=io.storage_view(value,[d.batch,width])
        for begin in range(0,width,32768):
            raw=io.finite_f32_bytes(io.slice(flat,begin,[d.batch,min(32768,width-begin)]));assert raw==b'\0'*len(raw)
    evidence['initial_expansion_failure_reset_zeroed']=True
    fresh_prefill(model,case);v=s.verifier;before=state(model);v.begin()
    lifecycle={'parent_reset_during_draft':reject(d.reset),'parent_close_during_draft':reject(d.close),
        'ordinary_chunk_during_draft':reject(lambda:d.chunk_hidden([],[],[],False)),
        'prediction_before_verification':reject(v.predictions),'second_begin':reject(v.begin),
        'target_program_close_during_draft':reject(v.program.close),'mtp_tree_close_during_draft':reject(s.tree_mtp.close)}
    assert v.phase=='drafting' and v.pending and not d.poisoned
    v.rollback();assert state(model)==before and not d.poisoned and not v.pending
    lifecycle['stale_predictions_after_rollback']=reject(v.predictions);evidence['draft_rollback_lifecycle']=lifecycle
    # Indexed query gathering, immutable candidate basket and owner epochs.
    v.begin();draft_before=state(model)
    stale=reject(lambda:s.tree_mtp.candidates_indexed_many([[0]*d.batch],v,s.candidates,2))
    before_ids=garnet.tensor_to_cpu(s.candidates).tolist()
    proposal=s.tree_mtp.draft(s,s.current,[1]*d.batch,[9]*d.batch,s.current_positions,s.stops,v,s.beam)
    assert proposal['greedy_spine_preserved'] and proposal['proposal_priority']=='anchored_binary_top2_conditional_nll'
    spine=[0]*d.batch;done=[False]*d.batch;spine_nodes=[[] for _ in range(d.batch)]
    for level in range(1,8):
        nodes=[0 if done[row] else spine[row] for row in range(d.batch)]
        candidates,costs=s.tree_mtp.candidates_indexed_many([nodes],v,s.candidates,2,True)
        for row in range(d.batch):
            if done[row]:continue
            token=candidates[0][row][0]
            matches=[node for node in range(1,proposal['used_nodes'][row]) if proposal['parents'][row][node]==spine[row] and proposal['depth'][row][node]==level and proposal['tokens'][row][node]==token]
            assert len(matches)==1,'Scored pruning discarded the live MTP greedy spine'
            spine[row]=matches[0];spine_nodes[row].append(matches[0]);done[row]=token in s.stops
        if all(done):break
    evidence['greedy_spine_verified_against_live_MTP_heads']=spine_nodes
    tp=s.tree_mtp;feature_bytes=row_major_bytes(io,tp.features,d.batch,tp.nodes*s.hidden_size);h=s.hidden_size

    results=[];f=s.beam
    for base in range(0,tp.nodes,f):
        nodes=[[(row*3+base+offset)%tp.nodes for row in range(d.batch)] for offset in range(f)]
        values,costs=tp.candidates_indexed_many(nodes,v,s.candidates,2,True)
        assert tp.candidates_indexed_many(nodes,v,s.candidates,2,True)==(values,costs)
        assert tp.candidates_indexed_many(nodes,v,s.candidates,2)==values
        assert tp.candidates_indexed_many(list(reversed(nodes)),v,s.candidates,2,True)==(list(reversed(values)),list(reversed(costs)))
        actual=io.finite_f32_bytes(tp.indexed_frontier_features)
        for row in range(d.batch):
            for offset,node_rows in enumerate(reversed(nodes)):
                source=(row*tp.nodes+node_rows[row])*h*4;dest=(row*f+offset)*h*4
                assert actual[dest:dest+h*4]==feature_bytes[source:source+h*4]
        results.append(values)
    assert garnet.tensor_to_cpu(s.candidates).tolist()==before_ids
    before_buffer=io.finite_f32_bytes(tp.indexed_frontier_features);invalid=[]
    for bad in ([],[[0]*d.batch]*(f+1),[[tp.nodes]*d.batch],[[-1]*d.batch],[[True]*d.batch],[[0]*d.batch,[tp.nodes]*d.batch]):
        invalid.append(reject(lambda:tp.candidates_indexed_many(bad,v,s.candidates,2)))
        assert io.finite_f32_bytes(tp.indexed_frontier_features)==before_buffer
    invalid.append(reject(lambda:tp.candidates_indexed_many([[0]*d.batch],v,None,2)))
    for bad_count in (0,3,True):
        invalid.append(reject(lambda:tp.candidates_indexed_many([[0]*d.batch],v,s.candidates,bad_count)))
        assert io.finite_f32_bytes(tp.indexed_frontier_features)==before_buffer
    assert state(model)==draft_before
    v.rollback();assert state(model)==draft_before
    released=reject(lambda:tp.candidates_indexed_many([[0]*d.batch],v,s.candidates,2))
    evidence['indexed_frontier_protocol_controls']={'selected_feature_gather_exact':True,'queries_replay_and_reverse_order_exact':True,
        'candidate_basket_immutable':True,'invalid_before_write':invalid,'stale_epoch_rejected':stale,'released_owner_rejected':released}
    # Failure after completed target/MTP commit and real shortlist refresh.
    refresh_before=state(model);real_refresh=s._refresh_candidates;refresh_called=[]
    def injected_refresh(*args):
        real_refresh(*args);refresh_called.append(True)
        assert not v.pending and d.state_transaction is None and d.slots==s.slots
        raise RuntimeError('INJECTED_AFTER_REAL_SHORTLIST_REFRESH')
    s._refresh_candidates=injected_refresh
    try:refresh_error=reject(lambda:s.generate(32))
    finally:s._refresh_candidates=real_refresh
    assert refresh_called and 'INJECTED_AFTER_REAL_SHORTLIST_REFRESH' in refresh_error
    assert d.poisoned and s.poisoned and not v.pending and d.state_transaction is None and state(model)!=refresh_before
    evidence['after_real_shortlist_refresh_poison']={'reason':refresh_error,'new_generate_rejected':reject(lambda:s.generate(32))}
    s.reset();assert not d.poisoned and not s.poisoned and d.slots==s.slots==[0]*d.batch and s.verifier.last_commit is None
    for value,width in ((d.state,d.programs[d.decode_chunk_size].state_width),(d.last_hidden,s.hidden_size),(s.key,s.cache_width),(s.value,s.cache_width),(s.last_hidden,s.hidden_size)):
        flat=io.storage_view(value,[d.batch,width])
        for begin in range(0,width,32768):
            raw=io.finite_f32_bytes(io.slice(flat,begin,[d.batch,min(32768,width-begin)]));assert raw==b'\0'*len(raw)
    evidence['shortlist_refresh_failure_reset_zeroed']=True
    fresh_prefill(model,case)
    save()
    target_before=state(model);mtp_before={name:digest_tensor(io,getattr(s,name),d.batch,s.cache_width if name in ('key','value') else s.hidden_size) for name in ('key','value','last_hidden')}
    real_correct=s._correct_tree;completion_controls=[]
    def injected(*args):
        real_correct(*args)
        assert v.pending and v.phase=='completing' and d.state_transaction is v and d.poisoned
        completion_controls.append({'rollback_rejected':reject(v.rollback),'verifier_close_rejected':reject(v.close),'parent_reset_rejected':reject(d.reset),
            'second_commit_rejected':reject(lambda:v.commit([-1]*d.batch)),'prediction_rejected':reject(v.predictions)})
        errors=[]
        def foreign():
            try:d.reset()
            except Exception as error:errors.append(str(error))
        thread=threading.Thread(target=foreign);thread.start();thread.join(10);assert not thread.is_alive() and len(errors)==1
        completion_controls[-1]['foreign_parent_reset_rejected']=errors[0]
        raise RuntimeError('INJECTED_AFTER_REAL_TARGET_AND_MTP_CORRECTION')
    s._correct_tree=injected
    try:reason=reject(lambda:s.generate(8))
    finally:s._correct_tree=real_correct
    assert 'INJECTED_AFTER_REAL_TARGET_AND_MTP_CORRECTION' in reason and completion_controls
    assert d.poisoned and s.poisoned and not v.pending and d.state_transaction is None and state(model)['canonical_sha256']!=target_before['canonical_sha256']
    mtp_after={name:digest_tensor(io,getattr(s,name),d.batch,s.cache_width if name in ('key','value') else s.hidden_size) for name in ('key','value','last_hidden')}
    assert mtp_before!=mtp_after,'Injected failure must follow real shifted MTP correction writes'
    evidence['after_real_target_and_mtp_write_poison']={'reason':reason,'completion_controls':completion_controls,
        'target_before':target_before,'target_after':state(model),'mtp_before':mtp_before,'mtp_after':mtp_after,
        'new_generate_rejected':reject(lambda:s.generate(8)),'new_target_begin_rejected':reject(v.begin)}
    s.reset();assert not s.poisoned and not d.poisoned and d.slots==s.slots==[0]*d.batch
    for value,width in ((d.state,d.programs[d.decode_chunk_size].state_width),(d.last_hidden,s.hidden_size),(s.key,s.cache_width),(s.value,s.cache_width),(s.last_hidden,s.hidden_size)):
        flat=io.storage_view(value,[d.batch,width])
        for start in range(0,width,32768):
            raw=io.finite_f32_bytes(io.slice(flat,start,[d.batch,min(32768,width-start)]));assert raw==b'\0'*len(raw)
    evidence['explicit_reset_zeroes_both_models_state_hidden_and_clocks']=True
    assert all(x['result']['speculative_metrics']['proposal_priority']=='anchored_binary_top2_conditional_nll' for x in evidence['tree'])
    evidence['scoped_owned_memory']=model.memory_info();evidence['all_lifecycle_controls_passed']=True;save()
    assert not evidence['failed_quality_gates'],'Original full-generation output/state exact gate failed; retain every result, do not relax'
    evidence['all_passed']=True;save()
except Exception as error:
    evidence['error']=repr(error);save();raise
finally:
    specmod.CUDASpeculativeSession=saved_factory
    if model is not None:model.close()
    event('closed')
