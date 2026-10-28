"""Three exact replays of a frozen fast-control case; third is profiled only."""
import ctypes,hashlib,json,os,sys,threading,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model

request_path,output_path=map(Path,sys.argv[1:3]);assert not output_path.exists()
request=json.loads(request_path.read_text(encoding='utf-8-sig'));case=request['case'];expected=request['expected_result']
assert os.environ['GARNET_QWEN35_FIXED_RMS']=='0'
assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
p=request['profile']
assert p['batch']==1 and p['speculativeMTP']==3 and p['speculativeHeadCandidates']==8192
assert p['prefillChunk']==64 and p['decodeChunk']==4 and not p.get('internalPackedStores',False)
bridge=ctypes.CDLL(request['nvtx_bridge']);bridge.qwen35NvtxPush.argtypes=[ctypes.c_char_p];bridge.qwen35NvtxPush.restype=ctypes.c_int
assert bridge.qwen35NvtxVersion()==3
cuda=ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
bridge.qwen35NvtxPush(b'fast_profile_bootstrap');bridge.qwen35NvtxPop()
model=None;records=[];restore=[];stack=[];stages={};thread=None;marks=[]
keys=('prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','visual_tokens_per_row')

def push(label):
    assert threading.get_ident()==thread
    # Retain the existing parser namespace; this worker performs no tree inference.
    bridge.qwen35NvtxPush(('garnet_tree:'+label).encode('ascii'));stack.append(label);marks.append({'operation':'push','label':label})

def pop(label):
    assert stack.pop()==label
    bridge.qwen35NvtxPop();marks.append({'operation':'pop','label':label})

def patch(obj,name,label):
    old=getattr(obj,name);assert callable(old) and hasattr(obj,'__dict__')
    restore.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
    def observed(*args,**kwargs):
        resolved=label(*args,**kwargs) if callable(label) else label
        push(resolved);start=time.perf_counter_ns()
        try:return old(*args,**kwargs)
        finally:
            elapsed=time.perf_counter_ns()-start;pop(resolved)
            row=stages.setdefault(resolved,{'calls':0,'inclusive_host_ns':0})
            row['calls']+=1;row['inclusive_host_ns']+=elapsed
    setattr(obj,name,observed)

def mtp_phase(flags,positions,predict,stage):
    assert stage in ('prefill','draft','correction')
    if stage=='prefill':return 'mtp_initial_draft' if predict else 'mtp_prompt_cache'
    return 'mtp_'+stage+'_execution'

try:
    model=Model(request['checkpoint'],p,request['model_id'])
    for i in range(3):
        capture=i==2;started=False
        if capture:
            thread=threading.get_ident();s=model.speculative;v=s.verifier
            assert s.depth==3 and s.head_candidates==8192
            for graph in [x.graph for x in model.decoder.programs.values()]+[model.decoder.head,s.program.graph,v.graph]:
                info=json.loads(graph.stats_json())
                assert info['contexts']==1 and info['graph_captures']>0 and info['graph_replays']>0 and info['graph_capture_failures']==0
            patch(model,'_prepare','input_row');patch(s,'reset','reset');patch(s,'prefill','prefill')
            patch(model.decoder,'prefill_sequence','target_prefill')
            patch(model.decoder,'chunk_hidden',lambda *a,**k:'target_prefill_chunk' if model.decoder.prefill_mode else 'target_verify_chunk')
            patch(s,'_call',mtp_phase);patch(s,'_draft','draft_prepare');patch(s,'_correct','correction_prepare')
            patch(s,'generate','generation');patch(v,'verify','target_verify');patch(v,'commit_prefix','target_commit')
            patch(v,'_read','target_head_read');patch(v,'_write_native','target_state_prefix_commit');patch(v,'_write_canonical','target_state_full_commit')
        try:
            if capture:
                assert cuda.cudaProfilerStart()==0;started=True;push('request')
            try:result=model.infer_batch(case['prompts'],case['images'],32)
            finally:
                if started:pop('request');assert not stack
        finally:
            if started:
                try:assert cuda.cudaDeviceSynchronize()==0
                finally:assert cuda.cudaProfilerStop()==0
            for obj,name,existed,value in reversed(restore):
                if existed:setattr(obj,name,value)
                else:delattr(obj,name)
            restore=[]
        assert all(result[k]==expected[k] for k in keys)
        assert result['speculative_metrics']==expected['speculative_metrics']
        records.append({'request_index':i,'capture':capture,'result':result})
        print('FAST_PROFILE_REQUEST_COMPLETE',i,flush=True)
    m=records[-1]['result']['speculative_metrics']
    assert stages['target_verify']['calls']==m['target_verification_calls']
    assert stages['mtp_draft_execution']['calls']==m['mtp_draft_calls']
    assert stages['mtp_correction_execution']['calls']==m['mtp_correction_calls']
    assert stages['mtp_prompt_cache']['calls']+stages['mtp_initial_draft']['calls']==m['mtp_prefill_calls']
    assert stages['target_head_read']['calls']==m['target_head_evaluations']+m['target_head_reuses']
    output_path.write_text(json.dumps({'records':records,'stages':stages,'nvtx_events':marks,'process_id':os.getpid(),
        'all_exact_outputs_and_metrics':True,'scope':'Current fast2B B1 norm0 MTP3 control; two fresh-request warmups then one nested NVTX/Nsight capture with synchronized tail. Exact outputs/work against archived case. Diagnostic only, no primary latency/memory/causal-speed acceptance.',
        'full_goal_met':False},indent=2),encoding='utf-8')
finally:
    if model is not None:model.close()
