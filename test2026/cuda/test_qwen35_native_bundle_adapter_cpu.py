"""CPU-only original-AST/dispatch controls with fake native objects; no GPU imports."""
import ast,importlib.util,json,sys,types
from pathlib import Path
root=Path(sys.argv[1]).resolve();repo=root/'Garnet';directory=Path(__file__).resolve().parent
output=root/'qwen35-native-bundle-adapter-cpu-v1.json';assert not output.exists()
path=repo/'xModel/qwen35/vl_0_8b/cuda_speculative.py'
tree=ast.parse(path.read_text())
klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CUDASpeculativeSession')
call=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_call')
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[call],type_ignores=[])),str(path),'exec'),ns)
def close_base(self):self.events.append('base_close')
StubBase=type('CUDASpeculativeSession',(),{'_call':ns['_call'],'close':close_base})
base=types.ModuleType('xModel.qwen35.vl_0_8b.cuda_speculative');base.__file__=str(path);base.CUDASpeculativeSession=StubBase
for name in ('xModel','xModel.qwen35','xModel.qwen35.vl_0_8b'):
    module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
sys.modules['xModel.qwen35.vl_0_8b.cuda_speculative']=base
sys.modules['xModel.qwen35.vl_0_8b'].cuda_speculative=base
class IO:
    def __init__(self):self.controls=[];self.posts=[]
    def update_request(self,a,p,s,flags,positions,slots):self.controls.append([flags,positions,slots])
    def slice_into(self,packed,start,dst):self.posts.append(['slice',start,dst])
    def copy_last_chunk(self,hidden,active,last):self.posts.append(['last',hidden,active,last])
class Graph:
    def __init__(self,owner):self.owner=owner
    def stats_json(self):return json.dumps({'contexts':1,'graph_capture_failures':0,'graph_captures':int(self.owner.normal_calls>=2)})
class Program:
    def __init__(self,owner):self.owner=owner;self.normal_calls=0;self.work=0;self.graph=Graph(self)
    def run(self,args):self.normal_calls+=1;self.work+=1;return 'packed'
class Bundle:
    def __init__(self):self.commands=[];self.submissions=0;self.closed=False
    def add_graph(self,graph,args):self.graph=graph;return 'packed'
    def slice_into(self,packed,start,dst):self.commands.append(('slice',packed,start,dst))
    def copy_last_chunk(self,h,a,l):self.commands.append(('last',h,a,l))
    def seal(self):pass
    def run(self):
        self.submissions+=1;self.graph.owner.work+=1
        io=self.graph.owner.owner.io
        for kind,*args in self.commands:
            (io.slice_into if kind=='slice' else io.copy_last_chunk)(*args)
    def close(self):self.closed=True;self.graph.owner.owner.events.append('bundle_close')
    def stats_json(self):return json.dumps({'submissions':self.submissions,'closed':self.closed})
fake=types.ModuleType('garnet');fake.cuda_execution_bundle=Bundle;sys.modules['garnet']=fake
spec=importlib.util.spec_from_file_location('qwen35_native_bundle_experiment_cpu',directory/'qwen35_native_bundle_experiment.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for args,kwargs in [([] ,{'gpu_draft_bundle':True}),([None]*6+[True],{})]:
    try:module.NativeBundleMTPSession(*args,**kwargs)
    except ValueError:pass
    else:raise AssertionError('Competing native draft ownership must reject before construction')
def session(cls):
    s=object.__new__(cls);s.batch=1;s.size=4;s.hidden_size=8;s.cache_width=16;s.capacity=32;s.slots=[0];s.head_candidates=0
    s.decoder=types.SimpleNamespace(resident_controls=True);s.io=IO();s.program=Program(s);s.events=[]
    for name in ('target','embedding','key','value','position','slot_tensor','active_tensor','hidden','last_hidden'):setattr(s,name,name)
    s.metrics={'mtp_prefill_calls':0,'mtp_head_calls':0};s.native_bundle=None;s.bundle_submissions=0;s.bundle_fallback_calls=0;s.poisoned=False;s.draft_runner=None
    return s
original=session(StubBase);candidate=session(module.NativeBundleMTPSession)
for index in range(3):
    positions=[[[index,index,index]]]
    assert original._call([[1]],positions,False,'prefill') is None
    assert candidate._call([[1]],positions,False,'prefill') is None
assert original.slots==candidate.slots==[3]
assert original.metrics==candidate.metrics=={'mtp_prefill_calls':3,'mtp_head_calls':0}
assert original.io.controls==candidate.io.controls and original.io.posts==candidate.io.posts
assert original.program.work==candidate.program.work==3
assert candidate.program.normal_calls==2 and candidate.bundle_fallback_calls==2 and candidate.bundle_submissions==1
candidate.close();assert candidate.events==['bundle_close','base_close']
invalid=session(module.NativeBundleMTPSession);invalid.capacity=0
try:invalid._call([[1]],[[[0,0,0]]],False,'prefill')
except ValueError:pass
else:raise AssertionError('Original capacity check bypassed')
assert invalid.program.work==0 and not invalid.io.controls and not invalid.io.posts
with output.open('x') as stream:json.dump({'all_controls_passed':True,'ast_proof':module.AST_PROOF,
    'original_control_values_slots_metrics_and_dispatch_effects_equal':True,'three_calls_three_model_evaluations':True,
    'two_actual_fallback_calls_before_capture_then_bundle':True,'capacity_failure_before_any_work':True,'bundle_closes_before_parent':True,
    'scope':'CPU fake-object adapter controls only; no real garnet/torch/CUDA imported, no tensors or native performance/quality/memory acceptance.'},stream,indent=2)
print('NATIVE_BUNDLE_ADAPTER_CPU_CONTROLS_PASS')
