"""Unwired native MTP dispatch experiment; never changes public serving defaults.

Only the old run/three slices/last-chunk dispatch is replaced. All host slot,
position, active, capacity, proposal, verification and commit logic is retained.
The first two actual calls warm/capture the old graph; no extra model pass.
"""
import ast
import hashlib
import json
from pathlib import Path
import garnet
from xModel.qwen35.vl_0_8b import cuda_speculative as base

def dispatch(self):
    args=[self.target,self.embedding,self.key,self.value,self.position,self.slot_tensor,self.active_tensor]
    if self.native_bundle is not None:
        self.native_bundle.run()
        self.bundle_submissions+=1
        return
    packed=self.program.run(args)
    self.io.slice_into(packed,0,self.hidden)
    self.io.slice_into(packed,self.size*self.hidden_size,self.key)
    self.io.slice_into(packed,self.size*self.hidden_size+self.cache_width,self.value)
    self.io.copy_last_chunk(self.hidden,self.active_tensor,self.last_hidden)
    self.bundle_fallback_calls+=1
    stats=json.loads(self.program.graph.stats_json())
    assert stats['contexts']==1 and stats['graph_capture_failures']==0
    if stats['graph_captures']==0:return
    assert stats['graph_captures']==1
    bundle=garnet.cuda_execution_bundle()
    try:
        output=bundle.add_graph(self.program.graph,args)
        bundle.slice_into(output,0,self.hidden)
        bundle.slice_into(output,self.size*self.hidden_size,self.key)
        bundle.slice_into(output,self.size*self.hidden_size+self.cache_width,self.value)
        bundle.copy_last_chunk(self.hidden,self.active_tensor,self.last_hidden)
        bundle.seal()
        self.native_bundle=bundle
    except Exception:
        bundle.close();self.poisoned=True
        raise

class NativeBundleMTPSession(base.CUDASpeculativeSession):
    def __init__(self,*args,**kwargs):
        self.native_bundle=None
        self.bundle_submissions=0
        self.bundle_fallback_calls=0
        super().__init__(*args,**kwargs)
    def bundle_info(self):
        return {'submissions':self.bundle_submissions,'fallback_calls':self.bundle_fallback_calls,
                'bundle':json.loads(self.native_bundle.stats_json()) if self.native_bundle is not None else None,
                'scope':'Opt-in experiment; no serving-default, primary latency/memory or2x acceptance.'}
    def close(self):
        try:
            if self.native_bundle is not None:self.native_bundle.close()
        finally:
            self.native_bundle=None
            super().close()

source_path=Path(base.__file__).resolve()
source=source_path.read_text()
tree=ast.parse(source)
classes=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='CUDASpeculativeSession']
assert len(classes)==1
methods=[x for x in classes[0].body if isinstance(x,ast.FunctionDef) and x.name=='_call']
assert len(methods)==1
original=methods[0]
expected=ast.parse('''packed=self.program.run([self.target,self.embedding,self.key,self.value,self.position,self.slot_tensor,self.active_tensor])
self.io.slice_into(packed,0,self.hidden)
self.io.slice_into(packed,self.size*self.hidden_size,self.key)
self.io.slice_into(packed,self.size*self.hidden_size+self.cache_width,self.value)
self.io.copy_last_chunk(self.hidden,self.active_tensor,self.last_hidden)
''').body
dump=lambda nodes:ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
hits=[i for i in range(len(original.body)-len(expected)+1) if dump(original.body[i:i+len(expected)])==dump(expected)]
assert len(hits)==1,'Original MTP dispatch shape changed; do not patch by approximation'
index=hits[0]
before=dump(original.body[:index]);after=dump(original.body[index+len(expected):])
original.body=original.body[:index]+ast.parse('_native_bundle_dispatch(self)').body+original.body[index+len(expected):]
assert dump(original.body[:index])==before and dump(original.body[index+1:])==after
assert original.decorator_list==[] and original.returns is None
namespace=dict(base.CUDASpeculativeSession._call.__globals__)
namespace['_native_bundle_dispatch']=dispatch
exec(compile(ast.fix_missing_locations(ast.Module(body=[original],type_ignores=[])),str(source_path)+'::native-bundle-experiment','exec'),namespace)
NativeBundleMTPSession._call=namespace['_call']
AST_PROOF={'original_source':str(source_path),'original_source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
           'replaced_statements':5,'original_dispatch_ast_sha256':hashlib.sha256(dump(expected).encode()).hexdigest(),
           'all_prefix_and_suffix_statements_unchanged':True,'extra_model_passes':0,
           'not_yet_qualified':True}
