"""One resident production2B fast model, four identical28-request periods; diagnostic only.

No correctness-state downloads, profiler, weight scans or process-memory sampler
run inside this timing worker. Allocation ledgers are read after each API timer.
"""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from xModel.qwen35.vl_0_8b.cuda_tree_speculative import CUDATreeSpeculativeSession
from qwen35_benchmark_workload import batch_inputs,input_hashes
from qwen35_tree_parent_reservation import reserve_tree_parent
from qwen35_stream_evidence import StreamEvidence

request_path,output_path=map(Path,sys.argv[1:3])
assert not output_path.exists(),'Preserve original cohort'
request=json.loads(request_path.read_text(encoding='utf-8-sig'))
assert os.environ['GARNET_QWEN35_FIXED_RMS']==str(request['fixed_rms'])
profile=request['profile'];batch=profile['batch'];tree=request['kind']=='tree_abba'
assert batch==1 and not tree and request['rounds']==7 and request['limit']==32 and request['fixed_rms']==0
evidence={'request':request,'completed':False,'error':None,
    'full_goal_met':False,'timing_scope':'API-return Model.infer_batch, including reset/JPEG/vision/tokenize/prefill/decode/detokenize; load excluded. No explicit final GPU-tail synchronization beyond API operations.',
    'process_memory_measured':False,'complete_state_correctness_measured':False,'primary_performance_eligible':False,'diagnostic_purpose':'Long-lived API/owned-allocation/capture stability with append-once bounded evidence; coarse Windows threadCPU charge is quantized, not a pure GPU/CPU split.'}

journal=StreamEvidence(output_path,evidence)
def event(name,**values):
    row={'name':name,'pid':os.getpid(),'unix_seconds':time.time()};row.update(values)
    journal.event(row);print(json.dumps(row),flush=True)
def captured(graph):
    assert graph['contexts']==1 and graph['graph_captures']>0 and graph['graph_replays']>0 and graph['graph_capture_failures']==0,graph

model=None;saved_factory=specmod.CUDASpeculativeSession;completed=False;error_text=None
try:
    if tree:
        def factory(*args):return CUDATreeSpeculativeSession(*args,nodes=request['nodes'],beam=request['beam'],share_parent_storage=True,batch_frontier_heads=False,batch_indexed_heads=True,scored_indexed_heads=False)
        specmod.CUDASpeculativeSession=factory
    event('load_begin')
    try:
        if tree:
            with reserve_tree_parent(request):model=Model(request['checkpoint'],profile,request['model_id'])
        else:model=Model(request['checkpoint'],profile,request['model_id'])
    finally:specmod.CUDASpeculativeSession=saved_factory
    event('load_complete')
    periods=[('fast_P1',False),('fast_P2',False),('fast_P3',False),('fast_P4',False)]
    frozen_owners=None
    for period_index,(policy_role,scored) in enumerate(periods):
        spec=model.speculative
        assert model.decoder.state_transaction is None and not model.decoder.poisoned and not spec.poisoned
        assert not spec.ready or spec.finished
        owners=[id(model),id(model.decoder),id(spec),id(model.vision),id(model.decoder.io)]+[id(p.graph) for p in model.decoder.programs.values()]
        if frozen_owners is None:frozen_owners=owners
        assert owners==frozen_owners
        journal.period_owner({'period':policy_role,'owner_ids':owners})
        event('period_begin',period=policy_role)
        for trial in range(7):
            for group in range(4):
                names,paths=batch_inputs(request['dataset'],trial,group,batch,True)
                hashes=input_hashes(paths)
                event('request_begin',period=policy_role,trial=trial,group=group)
                cpu_start=time.thread_time_ns()
                start=time.monotonic()
                result=model.infer_batch([request['prompt']]*batch,paths,32)
                elapsed=time.monotonic()-start
                cpu_elapsed=time.thread_time_ns()-cpu_start
                assert result['vision_compile_ms']==0 and result['visual_tokens_per_row']==[60]*batch
                assert all(n<=256 for n in result['prompt_tokens'])
                assert result['cache_policy']=={'prefix':False,'image':False,'answer':False}
                assert not model.decoder.poisoned and model.decoder.state_transaction is None
                if model.speculative is not None:assert model.speculative.slots==model.decoder.slots
                assert model.decoder.slots==[p+s-1 for p,s in zip(result['prompt_tokens'],result['sampled_tokens_including_stop'])]
                if profile['reuseVerifiedHead']:
                    metrics=result['speculative_metrics']
                    assert metrics['target_head_evaluations']==metrics['target_verification_calls']
                    assert metrics['target_head_reuses']==metrics['target_prefix_replays']
                memory=model.memory_info();io=json.loads(model.decoder.io.memory_json())
                assert memory['decoder']['compact_kv_outputs']
                assert all(x['copy_only'] and x['compact_kv_outputs'] for x in memory['decoder']['compact_state_commit_helpers'])
                unused=[]
                if trial>=2:
                    captured(memory['vision'])
                    if tree:
                        # programs are constructed in D8,P64 order. D8 is retained
                        # only as a borrowed owner/profile; tree never executes it.
                        assert list(model.decoder.programs)==[8,64]
                        assert memory['decoder']['graphs'][0]['graph_replays']==0
                        unused=['parent_D8_body']
                        for graph in memory['decoder']['graphs'][1:]:captured(graph)
                        spec=memory['speculative']
                        for graph in spec['mtp_graphs'].values():captured(graph)
                        captured(spec['wide_target_norm_graph'])
                        captured(spec['target_verifier']['tree']['graph'])
                        captured(spec['target_verifier']['tree']['norm_graph'])
                        captured(spec['recursive_mtp_tree']['graph'])
                        assert spec['draft_shortlist_count']==profile['speculativeHeadCandidates'] and spec['target_full_vocabulary_verification']
                    else:
                        for graph in memory['decoder']['graphs']:captured(graph)
                result.update({'trial':trial,'warmup':trial<2,'workload_group':group,'images':names,
                    'input_sha256':hashes,'wall_seconds':elapsed,'request_output_limit':32,
                    'policy_role':policy_role,'period_index':period_index,'calling_thread_cpu_ns':cpu_elapsed,'process_id':os.getpid(),'fixed_rms':request['fixed_rms'],'kind':request['kind'],
                    'io_memory':io,'owned_memory':memory,'unused_graphs':unused,
                    'timing_scope':evidence['timing_scope']})
                journal.row(result)
                event('request_complete',period=policy_role,trial=trial,group=group,wall_seconds=elapsed,generated_tokens=result['generated_tokens'])
        event('period_complete',period=policy_role)
    assert journal.row_count==112
    completed=True
except Exception as error:
    error_text=str(error);raise
finally:
    specmod.CUDASpeculativeSession=saved_factory
    try:
        if model is not None:model.close()
    except Exception as close_error:
        completed=False
        if error_text is None:error_text='Model.close: '+str(close_error)
        raise
    finally:
        event('closed')
        journal.finish(completed,error_text)
print('STREAMED_COHORT_COMPLETE',request['role'],flush=True)
