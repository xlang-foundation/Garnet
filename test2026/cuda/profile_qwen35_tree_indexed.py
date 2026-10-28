"""One warmed tree request with nested NVTX ranges; diagnostic only."""
import ctypes,hashlib,json,os,sys,threading,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from xModel.qwen35.vl_0_8b.cuda_tree_speculative import CUDATreeSpeculativeSession
request_path,output_path=map(Path,sys.argv[1:3]);assert not output_path.exists()
request=json.loads(request_path.read_text());case=request['case']
expected=request['expected_result']
assert os.environ['GARNET_QWEN35_FIXED_RMS']=='1'
assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
bridge=ctypes.CDLL(request['nvtx_bridge']);bridge.qwen35NvtxPush.argtypes=[ctypes.c_char_p];bridge.qwen35NvtxPush.restype=ctypes.c_int
assert bridge.qwen35NvtxVersion()==3
cuda=ctypes.CDLL('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/x64/cudart64_13.dll')
bridge.qwen35NvtxPush(b'tree_profile_bootstrap');bridge.qwen35NvtxPop()
model=None;saved=specmod.CUDASpeculativeSession;records=[];restore=[];stack=[];stages={};thread=None;marks=[]
keys=('prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','visual_tokens_per_row')
def push(label):
    assert threading.get_ident()==thread
    bridge.qwen35NvtxPush(('garnet_tree:'+label).encode('ascii'));stack.append(label);marks.append({'operation':'push','label':label})
def pop(label):
    assert stack.pop()==label
    bridge.qwen35NvtxPop();marks.append({'operation':'pop','label':label})
def patch(obj,name,label):
    old=getattr(obj,name);assert callable(old) and hasattr(obj,'__dict__')
    restore.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
    def observed(*args,**kwargs):
        push(label);start=time.perf_counter_ns()
        try:return old(*args,**kwargs)
        finally:
            end=time.perf_counter_ns();pop(label)
            row=stages.setdefault(label,{'calls':0,'inclusive_host_ns':0})
            row['calls']+=1;row['inclusive_host_ns']+=end-start
    setattr(obj,name,observed)
try:
    def factory(*args):return CUDATreeSpeculativeSession(*args,nodes=request['nodes'],beam=request['beam'],share_parent_storage=True)
    specmod.CUDASpeculativeSession=factory
    try:model=Model(request['checkpoint'],request['tree_profile'],request['model_id'])
    finally:specmod.CUDASpeculativeSession=saved
    for i in range(3):
        capture=i==2;started=False
        if capture:
            thread=threading.get_ident();s=model.speculative;v=s.verifier
            patch(model,'_prepare','input_row');patch(s,'reset','reset');patch(s,'prefill','prefill')
            patch(model.decoder,'chunk_hidden','parent_chunk');patch(s,'_call','mtp_prefill_or_correction')
            patch(s,'generate','generation');patch(s.tree_mtp,'draft','draft');patch(s.tree_mtp,'candidates_indexed','draft_candidate_head');patch(s,'_refresh_candidates','shortlist_refresh')
            patch(s.tree_mtp,'run','draft_tree_forward');patch(v,'verify_started','target_verify')
            patch(v.program,'run','target_tree_forward');patch(v.program,'predictions','target_head')
            patch(v,'_selection','accept_metadata');patch(v,'_commit','target_commit_and_correction')
            patch(s,'_correct_tree','mtp_correction')
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
        print('TREE_PROFILE_REQUEST_COMPLETE',i,flush=True)
    assert stages['draft_candidate_head']['calls']==records[-1]['result']['speculative_metrics']['recursive_mtp_candidate_heads']
    output_path.write_text(json.dumps({'records':records,'stages':stages,'nvtx_events':marks,'process_id':os.getpid(),'all_exact_outputs_and_metrics':True,
        'scope':'One warmed fresh-image request, nested NVTX and inclusive host timers. Profiler overhead; not primary latency or causal speedup evidence.',
        'full_goal_met':False},indent=2))
finally:
    specmod.CUDASpeculativeSession=saved
    if model is not None:model.close()
