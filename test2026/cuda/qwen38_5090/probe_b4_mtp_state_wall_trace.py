"""Untimed B4 correctness probe with synchronized nested diagnostic wall traces.

Run only after the original HTTP pilot is terminal and archived. This driver
changes Python references in this process only; no native/source patch or speed
admission. Parent intervals include children and diagnostic logging overhead.
"""
from pathlib import Path
import hashlib,json,os,runpy,sys,time
root,out=map(Path,sys.argv[1:3]);assert not out.exists();out.mkdir()
methods=root/'Garnet/test2026/cuda/qwen38_5090'
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(root/'Garnet'))
name='batch4-t4-mtp-host-embedding-http-pilot-attempt1'
workflow=json.loads((root/'results'/name/'workflow.json').read_text())
assert workflow['status'] in ('process_failed','completed_needs_independent_replay')
events=[json.loads(line) for line in (root/'results'/name/'ledger/commands.jsonl').read_text().splitlines()]
assert events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
assert (root/'results'/(name+'-evidence.zip')).is_file()
proof=json.loads((root/'results/shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1-verified.json').read_text())
assert proof['model_gate_passed'] and proof['strict_same_profile_target_state_hash_gate_passed']
assert hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest()==proof['library_sha256']==workflow['library_sha256']
for path,want in workflow['workspace_source_sha256'].items():assert hashlib.sha256((root/'Garnet'/path).read_bytes()).hexdigest()==want
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.cuda_packed_program import CUDAPackedDecoderProgram
from xModel.qwen35.vl_0_8b.cuda_packed_replay import CUDAPackedDeltaReplayVerifier
from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
originals=[];decoder=None;active=False;stack=[];trace=out/'stages.jsonl'
def patch(owner,name,method):
 originals.append((owner,name,getattr(owner,name)));setattr(owner,name,method)
constructor=CUDAPackedDecoder.__init__
def construct(self,*args,**kwargs):
 global decoder
 constructor(self,*args,**kwargs);decoder=self
patch(CUDAPackedDecoder,'__init__',construct)
def barrier():
 # Existing finite readback synchronizes the current native execution stream.
 value=decoder.io.storage_view(decoder.last_hidden,[1,1])
 assert len(bytes(decoder.io.finite_f32_bytes(value)))==4

def instrument(owner,name,label):
 method=getattr(owner,name)
 def wrapped(self,*args,**kwargs):
  if not active:return method(self,*args,**kwargs)
  barrier();started=time.perf_counter();parent=stack[-1] if stack else None;stack.append(label);error=None
  try:
   result=method(self,*args,**kwargs);barrier();return result
  except BaseException as failure:error=repr(failure);raise
  finally:
   ended=time.perf_counter();stack.pop()
   with trace.open('a',encoding='utf8') as output:output.write(json.dumps({'stage':label,'parent':parent,'depth':len(stack),'started':started,'ended':ended,'wall_seconds':ended-started,'error':error})+'\n')
 patch(owner,name,wrapped)
for owner,name,label in [(CUDAPackedDecoder,'chunk_hidden','target_chunk'),(CUDAPackedDecoderProgram,'run','target_program'),(CUDAPackedDeltaReplayVerifier,'commit_prefix','prefix_commit'),(CUDASpeculativeSession,'_draft','draft'),(CUDASpeculativeSession,'_correct','draft_correct')]:instrument(owner,name,label)
generate=CUDASpeculativeSession.generate
def generate_trace(self,*args,**kwargs):
 global active
 assert not active;active=True
 try:return generate(self,*args,**kwargs)
 finally:active=False
patch(CUDASpeculativeSession,'generate',generate_trace)
metadata={'status':'running','scope':'Synchronized nested native-stream wall diagnostics during MTP generate only. Four-byte barriers and logging alter scheduling; parent includes child intervals. Not kernel timing, uninstrumented latency, HTTP throughput or a speedup. Existing full-model ID/state correctness probe retained.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'prior_http_exit_code':workflow['exit_code']}
try:
 sys.argv=[str(methods/'probe_shared_mtp3_full_b4_t4_host_embedding_workspace.py'),str(root),'4',str(out/'correctness'),'host']
 runpy.run_path(sys.argv[0],run_name='__main__');metadata['status']='correctness_finished_needs_independent_replay'
except BaseException as failure:metadata.update(status='failed',error=repr(failure));raise
finally:
 for owner,name,method in reversed(originals):setattr(owner,name,method)
 (out/'trace-metadata.json').write_text(json.dumps(metadata,indent=2))
