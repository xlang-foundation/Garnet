"""CPU controls for observation equivalence, commit shape and complete trace replay."""
from pathlib import Path
import hashlib,importlib.util,json,struct,sys,types,unittest
from unittest.mock import patch
from .layer_boundary_replay import compare,validate,HIDDEN,LAYERS
M=Path(__file__).resolve().parent;R=M.parents[2];sha=lambda b:hashlib.sha256(b).hexdigest()

class Buffer:
    def __init__(self,shape):self.shape=shape;self.values=None
class IO:
    def zeros(self,shape):return Buffer(shape)
    def slice_into(self,source,start,target):target.values=source[start:start+target.shape[1]]
class Base:
    def __init__(self,*args,**kwargs):self.io=args[2];self.graph=True;self.output_width=17;self.hidden_width=3;self.weights=['weight']
    def run(self,inputs):
        if self.graph is None:raise ValueError('closed')
        if len(inputs)!=5:raise ValueError('inputs')
        self.seen=inputs;return list(range(self.output_width))
    def close(self):self.graph=None

def study():
    module=types.ModuleType('xModel.qwen35.vl_0_8b.cuda_packed_program')
    for name in ('T','Qwen35CUDAChunk','flat_concat','concat','packed_view'):setattr(module,name,None)
    module.CUDAPackedDecoderProgram=Base
    spec=importlib.util.spec_from_file_location('isolated_boundary_study',M/'layer_boundary_program.py');result=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{module.__name__:module}):spec.loader.exec_module(result)
    return result

def reports():
    raw={};reports=[]
    for tokens in (32,128):
        r=dict(tokens=tokens,prompt_ids=list(range(128)),same_profile_primary_equivalence_passed=True,performance_admitted=False,boundaries=[],boundary_runs=2*(128//tokens),repeat_complete_prefill_state_vectors_exact=True)
        for start in range(0,128,tokens):
            row=dict(start=start,count=tokens,layers={})
            for layer in LAYERS:
                blob=bytearray(tokens*HIDDEN*4)
                if tokens==128 and layer==3:struct.pack_into('<f',blob,(11*HIDDEN+99)*4,0.125)
                name=f'{tokens}-{start}-{layer}';raw[name]=bytes(blob);row['layers'][str(layer)]=dict(file=name,words=tokens*HIDDEN,sha256=sha(blob),repeat_sha256=sha(blob))
            r['boundaries'].append(row)
        reports.append(r)
    return reports,raw

class LayerBoundaryTests(unittest.TestCase):
    def test_exact_forward_derivation_keeps_original_math_barriers_and_guards(self):
        record=json.loads((M/'evidence/layer-boundary-program-derivation.json').read_bytes());original=(R/record['original_source']).read_bytes();assert sha(original)==record['original_source_sha256']
        src=original.decode();old=src[src.index('    def _forward(self):'):src.index('    def run(self,inputs):')]
        current=(M/'layer_boundary_program.py').read_bytes();assert sha(current)==record['target_sha256'];text=current.decode();new=text[text.index('    def _forward(self):'):text.index('    def run(self,inputs):')]
        for step in reversed(record['method_changes']):assert new.count(step['new'])==1;new=new.replace(step['new'],step['old'])
        self.assertEqual(new,old);self.assertEqual(sha(old.encode()),record['original_method_sha256'])
        probe=(M/'probe_layer_boundary.py').read_text();self.assertIn("same_state(report['ordinary_prefill'],expected['ordinary_prefill'],read,oldraw.__getitem__)",probe);self.assertIn('snapshots(report,read)==snapshots(expected,oldraw.__getitem__)',probe)
        self.assertIn("'admission_oracle_job':'mtp-prefill16-full4k-admission-attempt1'",probe)

    def test_wide_shared_storage_returns_exact_original_commit_prefix_and_readonly_inputs(self):
        cls=study().LayerBoundaryProgram;p=cls('checkpoint',{},IO(),1,4096,32);inputs=[object() for _ in range(5)];before=list(inputs)
        self.assertEqual(p.output_width,32);self.assertEqual(p.ordinary_output_width,17)
        out=p.run(inputs);self.assertEqual(out.shape,[1,17]);self.assertEqual(out.values,list(range(17)));self.assertEqual(inputs,before)
        for i,value in enumerate(p.boundary_buffers):self.assertEqual(value.values,list(range(17+i*3,20+i*3)))
        self.assertEqual(p.boundary_runs,1)
        with self.assertRaises(ValueError):p.run(inputs[:4])
        p.close();self.assertIsNone(p.primary_output);self.assertEqual(p.boundary_buffers,[])
        with self.assertRaises(ValueError):p.run(inputs)
        for b,c,t in ((2,4096,32),(1,512,32),(1,4096,4)):
            with self.assertRaises(ValueError):cls('checkpoint',{},IO(),b,c,t)

    def test_private_binding_restored_on_constructor_failure_and_T4_unchanged(self):
        s=study();parent=types.ModuleType('xModel.qwen35.vl_0_8b');module=types.ModuleType(parent.__name__+'.cuda_packed_decoder');module.CUDAPackedDecoderProgram=Base;parent.cuda_packed_decoder=module
        def factory():
            self.assertIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),1,4096,4),Base)
            self.assertNotIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),1,4096,4),s.LayerBoundaryProgram)
            self.assertIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),1,4096,128),s.LayerBoundaryProgram)
            raise RuntimeError('retained synthetic failure')
        with patch.dict(sys.modules,{parent.__name__:parent,module.__name__:module}):
            with self.assertRaises(RuntimeError):s.construct_with_boundary_program(factory)
        self.assertIs(module.CUDAPackedDecoderProgram,Base)

    def test_complete_array_replay_locates_mutation_and_rejects_corrupt_unqualified_or_nonfinite(self):
        (a,b),raw=reports();result=compare(a,b,raw.__getitem__,raw.__getitem__,128)
        self.assertEqual(result['complete_F32_words_compared'],128*HIDDEN*5)
        self.assertEqual(result['first_different_layer']['layer'],3);self.assertEqual(result['first_different_layer']['token'],11);self.assertEqual(result['first_different_layer']['dimension'],99)
        self.assertEqual(result['layers']['3']['changed_F32_words'],1);self.assertFalse(result['cause_established'])
        b['same_profile_primary_equivalence_passed']=False
        with self.assertRaises(AssertionError):validate(b,raw.__getitem__,128)
        b['same_profile_primary_equivalence_passed']=True;v=b['boundaries'][0]['layers']['4'];v['repeat_sha256']='0'*64
        with self.assertRaises(AssertionError):validate(b,raw.__getitem__,128)
        v['repeat_sha256']=v['sha256'];raw[v['file']]=b'x'+raw[v['file']][1:]
        with self.assertRaises(AssertionError):validate(b,raw.__getitem__,128)
        blob=bytearray(raw[v['file']]);struct.pack_into('<f',blob,0,float('inf'));raw[v['file']]=bytes(blob);v['sha256']=v['repeat_sha256']=sha(blob)
        with self.assertRaises(AssertionError):validate(b,raw.__getitem__,128)
