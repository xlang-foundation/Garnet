"""One frozen prospective profile, 28 serial cache-busting real-image requests.

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

request_path,output_path=map(Path,sys.argv[1:3])
assert not output_path.exists(),'Preserve original cohort'
request=json.loads(request_path.read_text(encoding='utf-8-sig'))
assert os.environ['GARNET_QWEN35_FIXED_RMS']==str(request['fixed_rms'])
profile=request['profile'];batch=profile['batch'];tree=request['kind']=='tree'
assert batch in (1,2,4,8) and request['rounds']==7 and request['limit']==32
evidence={'request':request,'events':[],'rows':[],'completed':False,'error':None,
    'full_goal_met':False,'timing_scope':'API-return Model.infer_batch, including reset/JPEG/vision/tokenize/prefill/decode/detokenize; load excluded. No explicit final GPU-tail synchronization beyond API operations.',
    'process_memory_measured':False,'complete_state_correctness_measured':False}

def save():output_path.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
def event(name,**values):
    row={'name':name,'pid':os.getpid(),'unix_seconds':time.time()};row.update(values)
    evidence['events'].append(row);save();print(json.dumps(row),flush=True)
def captured(graph):
    assert graph['contexts']==1 and graph['graph_captures']>0 and graph['graph_replays']>0 and graph['graph_capture_failures']==0,graph

model=None;saved_factory=specmod.CUDASpeculativeSession
try:
    if tree:
        def factory(*args):return CUDATreeSpeculativeSession(*args,nodes=request['nodes'],beam=request['beam'],share_parent_storage=True,batch_frontier_heads=False)
        specmod.CUDASpeculativeSession=factory
    event('load_begin')
    try:model=Model(request['checkpoint'],profile,request['model_id'])
    finally:specmod.CUDASpeculativeSession=saved_factory
    event('load_complete')
    for trial in range(7):
        for group in range(4):
            names,paths=batch_inputs(request['dataset'],trial,group,batch,True)
            hashes=input_hashes(paths)
            event('request_begin',trial=trial,group=group)
            start=time.monotonic()
            result=model.infer_batch([request['prompt']]*batch,paths,32)
            elapsed=time.monotonic()-start
            assert result['vision_compile_ms']==0 and result['visual_tokens_per_row']==[60]*batch
            assert all(n<=256 for n in result['prompt_tokens'])
            assert result['cache_policy']=={'prefix':False,'image':False,'answer':False}
            assert not model.decoder.poisoned and model.decoder.state_transaction is None
            assert model.speculative.slots==model.decoder.slots
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
                'process_id':os.getpid(),'fixed_rms':request['fixed_rms'],'kind':request['kind'],
                'io_memory':io,'owned_memory':memory,'unused_graphs':unused,
                'timing_scope':evidence['timing_scope']})
            evidence['rows'].append(result)
            event('request_complete',trial=trial,group=group,wall_seconds=elapsed,generated_tokens=result['generated_tokens'])
    assert len(evidence['rows'])==28
    evidence['completed']=True
except Exception as error:
    evidence['error']=str(error);save();raise
finally:
    specmod.CUDASpeculativeSession=saved_factory
    if model is not None:model.close()
    event('closed')
print('FROZEN_TREE_COHORT_COMPLETE',request['role'],flush=True)
