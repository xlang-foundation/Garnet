"""CPU-only source-lineage replay; never loads TensorRT, Garnet or CUDA."""
from pathlib import Path
import hashlib,json

def digest(raw):return hashlib.sha256(raw).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def function(text,signature):
 start=text.index(signature);a=text.index('{',start);level=1;i=a+1
 while level:
  level+=(text[i]=='{')-(text[i]=='}');i+=1
 return text[start:i]
def replay(root):
 root=Path(root);m=root/'test2026/cuda/qwen38_5090';e=m/'evidence';observed={}
 def source(name,expected):
  raw=(root/name).read_bytes();require(digest(raw)==expected,'Source hash differs: '+name)
  observed[name]=digest(raw);return raw.decode().replace('\r\n','\n')
 def transformed(record,target):
  text=source(record['source'],record['source_sha256'])
  for before,after,count in record['changes']:
   require(text.count(before)==count,'Derivation replacement count differs: '+target);text=text.replace(before,after)
  raw=(m/target).read_bytes();require(raw==text.encode(),'Derived bytes differ: '+target)
  require(digest(raw)==record['result_sha256'],'Derived hash differs: '+target);observed[target]=digest(raw)
 wrapper=json.loads((e/'bf16-low-batch-overlay-plugin-source-derivation.json').read_bytes())
 for name,record in wrapper['files'].items():transformed(record,name)
 probe=json.loads((e/'bf16-low-batch-plugin-cpu-probe-derivation.json').read_bytes())
 transformed(probe,'probe_bf16_low_batch_plugin_contract.cpp')
 adapter=json.loads((e/'bf16-low-batch-plugin-cpu-adapter-derivation.json').read_bytes())
 result='#include "bf16_kv_low_batch_candidate.h"\n#include "qwen38_bf16_kv_overlay.cuh"\nnamespace Garnet {\nnamespace { int unexpected=0;\n'
 for tag in ('Legacy','LowBatch'):
  record=adapter['source_functions'][tag];text=source(record['source'],record['source_sha256'])
  body=function(text,'bool valid(');require(body==record['valid_function'],'Validity source differs')
  renamed=body.replace('bool valid(','bool valid'+tag+'(',1)
  require(renamed==record['renamed_valid_function'],'Validity rename differs');result+=renamed+'\n'
 result+='}\n'
 for tag,signature in [('Legacy','size_t Qwen38BF16KVOverlayWorkspace('),('LowBatch','size_t Qwen38BF16KVLowBatchCandidateWorkspace(')]:
  record=adapter['source_functions'][tag];text=source(record['source'],record['source_sha256']);body=function(text,signature)
  require(body==record['workspace_function'] and body.count('valid(c)')==1,'Workspace source differs')
  renamed=body.replace('valid(c)','valid'+tag+'(c)');require(renamed==record['renamed_workspace_function'],'Workspace rename differs');result+=renamed+'\n'
 result+='int BF16LowBatchPluginCPUUnexpectedEnqueues() noexcept{return unexpected;}\n'
 for name in ('RunQwen38BF16KVOverlay','RunQwen38BF16KVLowBatchCandidate'):
  result+='int '+name+'(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int64_t,int64_t,cudaStream_t) noexcept{++unexpected;return 1;}\n'
 result+='}\n';raw=(m/'bf16_low_batch_plugin_cpu_operators.cpp').read_bytes()
 require(raw==result.encode() and digest(raw)==adapter['result_sha256'],'CPU adapter differs')
 observed['bf16_low_batch_plugin_cpu_operators.cpp']=digest(raw)
 build=json.loads((e/'bf16-low-batch-overlay-plugin-local-build.json').read_bytes())
 require(build['runtime_executed'] is False and build['gpu_used'] is False and build['remote_mutated'] is False and build['model_qualified'] is False and build['performance_admitted'] is False,'Compile receipt overstates admission')
 for name,expected in build['source_sha256'].items():
  path=Path(name);require(path.is_file() and digest(path.read_bytes())==expected,'Current compiled input differs: '+name)
 require({v.lower() for v in build['imports']}=={'nvinfer_10.dll','kernel32.dll'},'CPU imports differ')
 fixture=[name for name in build['source_sha256'] if name.endswith('bf16_kv_fixture_layout.h')];require(len(fixture)==1,'Compiled fixture identity differs')
 require(digest((Path(fixture[0]).parent/'bf16-low-batch-overlay-plugin-cpu.exe').read_bytes())==build['executable_sha256'],'Compiled executable differs')
 ledger=e/'bf16-low-batch-overlay-plugin-local-build-ledger';events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()]
 require(len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[0]['id']==events[1]['id'] and events[1]['exit_code']==0 and events[0]['kind']=='build','Actual build ledger is not a successful single compile')
 for name,expected in events[0]['artifact_sha256'].items():require(digest(Path(name).read_bytes())==expected,'Logged build input differs: '+name)
 rawlog=(ledger/events[1]['raw_output_file']).read_bytes();require(0<=events[1]['raw_start_offset']<events[1]['raw_end_offset']<=len(rawlog),'Build raw output offsets differ')
 return {'status':'b1_overlay_plugin_source_derivations_replayed','source_sha256':observed,'private_compile_link_passed':True,'runtime_contract_controls_executed':False,'gpu_enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'scope':'Literal wrapper/probe transformations and extracted pure host workspace adapter verified against current files and build receipt. No runtime/ABI, GPU/model or speed result.'}

if __name__=='__main__':print(json.dumps(replay(Path(__file__).resolve().parents[3]),indent=2))
