"""CPU-only localization of list/index boundary under the actual xlang host."""
import os,sys,threading
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from xModel.qwen35.vl_0_8b.cuda_tree_mtp import CUDATreeMTPProgram

class Stub:pass
class IO:
    def zero(self,*args):pass
    def update(self,*args):pass
    def copy_last_chunk(self,*args):pass
    def write_chunk_token(self,*args):pass
    def predict_indexed(self,*args):return [4]
    def predict_indexed_top2(self,*args):return [4,7]

p=Stub();owner=Stub();owner.phase='drafting'
p.head=object();p.poisoned=False;p.state_transaction=owner
p.transaction_thread=threading.get_ident();p.state_transaction_epoch=1
tp=object.__new__(CUDATreeMTPProgram);tp.parent=p;tp.graph=object();tp.io=IO()
tp.share_parent_storage=True;tp.features_ready=True;tp.features_epoch=1
tp.batch=1;tp.nodes=16;tp.config={'vocab_size':64};tp.indexed_frontier_width=1
for name in ('features','select_mask','selected','indexed_frontier_features','indexed_frontier_queries'):setattr(tp,name,object())
flat=list(tp.io.predict_indexed(None,None));print('CPU_LIST_KIND',type(flat),repr(flat),flat[0],flat[0:1],flush=True)
def trace(frame,event,arg):
    if event=='line' and frame.f_code.co_name=='candidates_indexed_many' and frame.f_lineno==192:
        print('HELPER_LOCALS',{k:(str(type(v)),repr(v)) for k,v in frame.f_locals.items() if k in ('flat','f','b','row','offset','count','start')},flush=True)
    return trace
if os.environ.get('FRONTIER_DEBUG_TRACE')=='1':sys.settrace(trace)
for count in (1,2):
    print('CPU_HELPER_BEGIN',count,flush=True)
    print('CPU_HELPER_RESULT',tp.candidates_indexed_many([[0]],owner,object(),count),flush=True)
print('CPU_HELPER_COMPLETE',flush=True)
sys.path.insert(0,str(Path(__file__).resolve().parent))
from xModel.qwen35.vl_0_8b import cuda_packed_decoder as packed
from qwen35_tree_parent_reservation import reserve_tree_parent
for kwargs in ({'shared_workspace_min_bytes':True},{'shared_workspace_min_bytes':-1},{'shared_workspace_min_bytes':(1<<30)+1},
               {'shared_output_min_bytes':True},{'shared_output_min_bytes':-1},{'shared_output_min_bytes':(1<<30)+1},
               {'shared_workspace_min_bytes':4},{'shared_output_min_bytes':4}):
    try:packed.CUDAPackedDecoder('UNOPENED',{},**kwargs)
    except ValueError:pass
    else:raise AssertionError('Invalid reservation must reject before checkpoint/CUDA initialization')
real=packed.CUDAPackedDecoder
def fake(*args,**kwargs):return kwargs
packed.CUDAPackedDecoder=fake
try:
    request={'profile':{'batch':1},'parent_storage_reservation':{'workspace_bytes':47939072,'output_bytes':30559600}}
    with reserve_tree_parent(request):
        assert packed.CUDAPackedDecoder(None,None,1)=={'shared_workspace_min_bytes':47939072,'shared_output_min_bytes':30559600}
    assert packed.CUDAPackedDecoder is fake
finally:packed.CUDAPackedDecoder=real
print('CPU_CONSTRUCTION_RESERVATION_ADMISSION_AND_RESTORE_PASS',flush=True)
