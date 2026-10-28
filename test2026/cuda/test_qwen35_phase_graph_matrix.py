"""Sequential old and phase graphs, entire state/hidden and transaction checks."""
import hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.cuda_packed_replay import CUDAPackedDeltaReplayVerifier
request= json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'));output=Path(sys.argv[2]);assert not output.exists()
runs=[];events=[];error=None
def event(name):
    events.append({'name':name,'unix_seconds':time.time(),'pid':os.getpid()});print('PHASE_GRAPH_EVENT',name,flush=True)
def digest(model,value,width):
    result=hashlib.sha256();batch=model.batch;value=model.io.storage_view(value,[batch,width])
    for start in range(0,width,max(1,1048576//batch)):
        count=min(width-start,1048576//batch);part=model.io.slice(value,start,[batch,count]);result.update(model.io.finite_f32_bytes(part))
    return result.hexdigest()
def run(phase):
    event('phase_begin' if phase else 'baseline_begin');snapshots=[];model=None;verifier=None
    record={'phase':phase,'snapshots':snapshots,'complete':False};runs.append(record)
    try:
        model=CUDAPackedDecoder(request['checkpoint'],request['config'],request['batch'],256,64,4,retain_linear_inputs=True,register_delta_128=True,flat_concat=True,deferred_state_commit=True,shared_output_storage=True,join_decode_input_projections=True,shared_activation_storage=True,internal_packed_stores=request['internal_stores'],compact_kv_outputs=True,phase_output_partition_layers=4 if phase else 0)
        model.resident_controls=True;model.serial_target_graph_reuse=True;model.native_prefix_commit=True;model.enable_cuda_graph(True)
        verifier=CUDAPackedDeltaReplayVerifier(model,request['checkpoint'],4,True)
        width=model.programs[4].state_width
        def snapshot(name,predictions):
            snapshots.append({'name':name,'predictions':list(predictions),'slots':list(model.slots),'state':digest(model,model.state,width),'hidden':hashlib.sha256(model.io.finite_f32_bytes(model.last_hidden)).hexdigest()})
            print('PHASE_GRAPH_SNAPSHOT',phase,name,flush=True)
        tokens=[request['steps'][t%len(request['steps'])]['tokens'][0] for t in range(69)]
        hidden=[model.io.embedding([token]*model.batch) for token in tokens]
        snapshot('prefill69',model.prefill_sequence(hidden,[[[t]*3 for t in range(69)] for _ in range(model.batch)],[69-row%3 for row in range(model.batch)]))
        for step in range(2):snapshot('normal_step'+str(step),model.step([tokens[step]]*model.batch,[int((row+step)%3!=2) for row in range(model.batch)],[[slot]*3 for slot in model.slots]))
        for accepted in (0,1,2,3,4,'mixed','ragged'):
            before=digest(model,model.state,width);submitted=[[tokens[t]]*model.batch for t in range(4)];limits=[4-row%5 for row in range(model.batch)] if accepted=='ragged' else [4]*model.batch;active=[[int(t<limits[row]) for row in range(model.batch)] for t in range(4)]
            positions=[[[slot+t]*3 for t in range(4)] for slot in model.slots]
            verifier.verify(submitted,active,positions);assert digest(model,model.state,width)==before,'Deferred verify mutated canonical state'
            counts=[min(limits[row],row%5) for row in range(model.batch)] if accepted in ('mixed','ragged') else [accepted]*model.batch
            snapshot('accepted'+str(accepted),verifier.commit_prefix(counts))
            assert model.state_transaction is None and not verifier.pending and not model.poisoned
        before=digest(model,model.state,width);oldslots=list(model.slots);saved=[digest(model,value,model.io.tensor_bytes(value)//(model.batch*4)) for value in model.verification_tensors()]
        verifier.verify([[tokens[t]]*model.batch for t in range(4)],[[1]*model.batch for _ in range(4)],[[[slot+t]*3 for t in range(4)] for slot in model.slots])
        verifier.rollback();assert digest(model,model.state,width)==before and model.slots==oldslots and model.state_transaction is None
        assert saved==[digest(model,value,model.io.tensor_bytes(value)//(model.batch*4)) for value in model.verification_tensors()]
        record['rollback_exact']=True;record['memory']=model.memory_info();record['verifier_memory']=verifier.memory_info()
        assert record['memory']['owned_decoder_bytes'] is not None and all(g['allocator_tracking_enabled'] for g in record['memory']['graphs'])
        graphs=record['memory']['graphs'];assert all(not g['network_output_aliases_input'] for g in graphs)
        assert all(g['graph_capture_failures']==0 for g in graphs)
        if phase:
            model.reset();part=model.phase_prefill_programs[1];original=part.commit_all
            def fail_after_write(*args):original(*args);raise RuntimeError('Injected after actual partition canonical write')
            part.commit_all=fail_after_write;failed=False
            try:model.prefill_sequence(hidden[:2],[[[t]*3 for t in range(2)] for _ in range(model.batch)])
            except RuntimeError as failure:failed='Injected after actual' in str(failure)
            finally:part.commit_all=original
            assert failed and model.poisoned and model.state_transaction is None
            model.reset();zero=model.io.zeros([model.batch,width]);assert digest(model,model.state,width)==digest(model,zero,width)
            record['partial_prefill_real_write_poison_zero_reset']=True
            model.step([tokens[0]]*model.batch,[1]*model.batch,[[0]*3 for _ in range(model.batch)])
            verifier.verify([[tokens[t]]*model.batch for t in range(4)],[[1]*model.batch for _ in range(4)],[[[slot+t]*3 for t in range(4)] for slot in model.slots])
            original_write=verifier._write_native
            def fail_commit(spans):original_write(spans);raise RuntimeError('Injected after actual full accepted replay write')
            verifier._write_native=fail_commit;failed=False
            try:verifier.commit_prefix([4]*model.batch)
            except RuntimeError as failure:failed='Injected after actual' in str(failure)
            finally:verifier._write_native=original_write
            assert failed and model.poisoned and model.state_transaction is None and not verifier.pending
            model.reset();assert digest(model,model.state,width)==digest(model,zero,width)
            record['full_accepted_replay_real_write_poison_zero_reset']=True
        record['complete']=True
    finally:
        if verifier is not None:verifier.close()
        if model is not None:model.close()
        event('phase_closed' if phase else 'baseline_closed')
event('started')
try:
    run(False);run(True)
    assert runs[0]['snapshots']==runs[1]['snapshots'],'Complete canonical/hidden/ID mismatch; preserve evidence without tolerance relaxation'
    assert runs[0]['memory']['projection_weight_pool_bytes']==runs[1]['memory']['projection_weight_pool_bytes']
    assert runs[1]['memory']['shared_output_pool_bytes']<runs[0]['memory']['shared_output_pool_bytes']
except Exception as failure:error=repr(failure);raise
finally:
    event('closed');output.open('x',encoding='utf-8').write(json.dumps({'all_passed':error is None and len(runs)==2 and all(x['complete'] for x in runs),'error':error,'runs':runs,'events':events,'scope':request['scope'],'real_model_qualified':False,'performance_measured':False,'full_goal_met':False},indent=2))
