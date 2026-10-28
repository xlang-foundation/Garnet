"""Isolated tree generator WDDM diagnostic; polling invalidates latency."""
import hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from runtime.cuda_serving import Model
from xModel.qwen35.vl_0_8b import cuda_speculative as specmod
from xModel.qwen35.vl_0_8b.cuda_tree_speculative import CUDATreeSpeculativeSession
request_path,output_path=map(Path,sys.argv[1:3]);mode=sys.argv[3]
assert mode in ('control','parent') and not output_path.exists()
request=json.loads(request_path.read_text());profile=dict(request['tree_profile'])
shared=mode=='parent';profile['sharePackedActivationStorage']=shared
reference=json.loads(Path(request['correctness_reference']).read_text())
expected=next(x for x in reference['tree'] if x['limit']==32)['result']
keys=('prompt_token_ids','generated_token_ids','sampled_tokens_including_stop','finish_reasons','stop_token_ids','visual_tokens_per_row')
events=[];model=None
def emit(event,**values):
    events.append({'event':event,'unix_seconds':time.time(),'process_id':os.getpid(),'model':request['model_id'],
        'engine':'garnet','profile':profile,'mode':mode,**values})
    output_path.write_text(json.dumps(events,indent=2)+'\n')
    print('TREE_PARENT_MEMORY_EVENT',event,os.getpid(),flush=True)
case=request['case']
assert [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in case['images']]==case['input_sha256']
baseline=garnet.cuda_tensor_io();word=baseline.zeros([1]);baseline.zero(word)
assert baseline.finite_f32_bytes(word)==b'\0'*4
del word;baseline.close();del baseline
emit('baseline_begin');time.sleep(10);emit('baseline_end')
saved=specmod.CUDASpeculativeSession
try:
    emit('load_begin')
    def factory(*args):return CUDATreeSpeculativeSession(*args,nodes=request['nodes'],beam=request['beam'],share_parent_storage=shared)
    specmod.CUDASpeculativeSession=factory
    try:model=Model(request['checkpoint'],profile,request['model_id'])
    finally:specmod.CUDASpeculativeSession=saved
    emit('loaded',owned_memory=model.memory_info())
    def infer(index):
        result=model.infer_batch(case['prompts'],case['images'],32)
        assert all(result[k]==expected[k] for k in keys)
        assert result['speculative_metrics']==expected['speculative_metrics']
        assert not model.decoder.poisoned and not model.speculative.poisoned and model.decoder.state_transaction is None
        return {'trial':index,'output_signature':{k:result[k] for k in keys},'speculative_metrics':result['speculative_metrics'],
            'owned_memory':model.memory_info()}
    for i in range(2):emit('warmup_complete',**infer(i))
    emit('steady_before_begin',owned_memory=model.memory_info());time.sleep(10);emit('steady_before_end')
    emit('inference_begin')
    for i in range(2,6):emit('request_complete',**infer(i))
    emit('inference_end')
    memory=model.memory_info()
    assert memory['speculative']['parent_storage_borrowed']==shared
    emit('steady_after_begin',owned_memory=memory);time.sleep(10);emit('steady_after_end')
finally:
    specmod.CUDASpeculativeSession=saved
    if model is not None:model.close()
    emit('closed_begin');time.sleep(10);emit('closed_end')
emit('complete')
