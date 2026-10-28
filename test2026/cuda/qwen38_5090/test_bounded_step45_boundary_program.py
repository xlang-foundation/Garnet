"""Exercise the actual new forward and actual flat_concat with native limit stub."""
from pathlib import Path
import ast,hashlib,json,types,unittest
M=Path(__file__).resolve().parent

class Tensor:
    def __init__(self,spans): self.spans=list(spans)
    def __mul__(self,op):
        if op[0] in ('reshape','qwen35_f32_barrier'): return self
        assert op[0]=='qwen35_concat_parts'
        if len(op[1]['parts'])>128: raise ValueError('native concat guard')
        return Tensor(self.spans+sum((v.spans for v in op[1]['parts']),[]))

def view(x,start,width,shape):
    if x is None:return Tensor([('unused-state',width)])
    result=[];cursor=0
    for name,count in x.spans:
        n=max(0,min(cursor+count,start+width)-max(cursor,start))
        if n:result.append((name,n))
        cursor+=count
    assert sum(n for _,n in result)==width
    return Tensor(result)

def actual_forward():
    calls=[]
    def unary(name,**kw):
        if name=='qwen35_concat_parts':calls.append(len(kw['parts'])+1)
        return name,kw
    env={'T':types.SimpleNamespace(fusion=lambda **kw:lambda f:f,unary_op=unary),'packed_view':view}
    raw=(M/'evidence/step45-frozen-cuda-packed-program.py.txt').read_bytes()
    audit=json.loads((M/'evidence/step45-layer-boundary-source-audit.json').read_bytes())
    assert hashlib.sha256(raw).hexdigest()==audit['frozen_sources']['candidate']['xModel/qwen35/vl_0_8b/cuda_packed_program.py']
    tree=ast.parse(raw)
    constructor=next(n for n in next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CUDAPackedDecoderProgram').body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assert "if retain_linear_inputs and kind == 'linear_attention':" in ast.unparse(constructor)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='flat_concat')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-flat-concat','exec'),env)
    def chunk(*args,**kw):
        layer=args[7];spans=[('h'+str(layer),20480),('s'+str(layer),2)]
        if layer%4!=3:spans.append(('p'+str(layer),1))
        return Tensor(spans)
    env.update(Qwen35CUDAChunk=chunk,concat=env['flat_concat'])
    tree=ast.parse((M/'bounded_step45_boundary_program.py').read_bytes());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)).body[0]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'actual-bounded-forward','exec'),env)
    return env,calls

class Contract(unittest.TestCase):
    def test_actual_forward_three_bounded_concats_preserve_all_primary_and_suffix_spans(self):
        for batch in (1,4):
            env,calls=actual_forward()
            layout=[dict(index=i,kind='full_attention' if i%4==3 else 'linear_attention',state1_width=1,state2_width=1,offset=i*2,state1_shape=[batch,1],state2_shape=[batch,1],payload_width=int(i%4!=3)) for i in range(64)]
            p=types.SimpleNamespace(config={'hidden_size':5120},batch=batch,size=4,layout=layout,flat_concat=True,delta_layout=None,weight_layers=[[] for _ in layout],external_compressed_weights=False,workspace=0,retain=True,register_delta_128=True,join_input_projections=False,joined_mlp_projections=True,boundary_layers=tuple(range(64)),kv_overlay_strategy=0)
            out=env['_forward'](p)(None,None,None,None,None)
            expected=[('h63',20480)]+[('s'+str(i),2) for i in range(64)]+[('p'+str(i),1) for i in range(64) if i%4!=3]+[('h'+str(i),20480) for i in range(64)]
            self.assertEqual(out.spans,expected);self.assertEqual(calls,[113,64,2]);self.assertTrue(all(n-1<=128 for n in calls))
            with self.assertRaisesRegex(ValueError,'native concat guard'):env['flat_concat']([Tensor([('x',1)]) for _ in range(177)])

    def test_arithmetic_AST_identical_after_removing_only_bounded_output_joins(self):
        def forward(name):
            c=next(n for n in ast.parse((M/name).read_bytes()).body if isinstance(n,ast.ClassDef))
            return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_forward')
        original=forward('decode_boundary_program.py');new=forward('bounded_step45_boundary_program.py')
        nested=next(n for n in new.body if isinstance(n,ast.FunctionDef))
        self.assertEqual([ast.unparse(n) for n in nested.body[-3:]],['primary = join([flat] + state_outputs + payload_outputs)','observations = join(boundary_outputs)','return join([primary, observations])'])
        nested.body[-3:]=[next(n for n in next(n for n in original.body if isinstance(n,ast.FunctionDef)).body if isinstance(n,ast.Return))]
        self.assertEqual(ast.dump(new,include_attributes=False),ast.dump(original,include_attributes=False))

    def test_exact_derivation_and_existing_native_limit_unchanged(self):
        p=json.loads((M/'evidence/bounded-step45-boundary-derivation.json').read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
        self.assertEqual(sha((M/p['source']).read_bytes()),p['source_sha256']);self.assertEqual(sha((M/p['target']).read_bytes()),p['target_sha256'])
        self.assertIn('item->second.Size()>128',p['native_guard_text']);self.assertEqual(p['maximum_extra_parts'],128);self.assertFalse(p['native_guard_or_precision_changed'])
        source=(M/p['target']).read_text(encoding='utf8');self.assertIn('else original)(*values,**options)',source);self.assertIn('finally:module.CUDAPackedDecoderProgram=original',source)

if __name__=='__main__':unittest.main()
