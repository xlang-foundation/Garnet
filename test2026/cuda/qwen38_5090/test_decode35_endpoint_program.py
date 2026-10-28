"""CPU topology and source controls for six final outputs; original native hook stays seven."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,types,unittest
from unittest.mock import patch
from .test_decode35_operand_program import isolated as original_isolated
from .test_decode_boundary_program import IO
M=Path(__file__).resolve().parent

def isolated():
    old,base=original_isolated();parent='test2026.cuda.qwen38_5090.'
    spec=importlib.util.spec_from_file_location(parent+'isolated_decode35_endpoint_program',M/'decode35_endpoint_program.py');module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{parent+'decode35_operand_program':old}):spec.loader.exec_module(module)
    return module,old,base

class EndpointContract(unittest.TestCase):
    def test_primary_all64_and_six_endpoint_rows(self):
        module,_,_=isolated()
        for batch in (1,4):
            p=module.Decode35EndpointProgram('checkpoint',{},IO(),batch,4096,4);start=17+64*20480
            self.assertEqual(p.output_width,start+102400);self.assertEqual(len(p.operand_buffers),6)
            primary=p.run([object()]*5);self.assertEqual(p.boundary_runs,1)
            for row in range(batch):
                base=row*10000000;self.assertEqual(list(primary.values[row]),list(range(base,base+17)))
                for layer,buffer in enumerate(p.boundary_buffers):self.assertEqual(buffer.values[row][0],base+17+layer*20480)
                cursor=start
                for (name,width),(recorded_name,offset,recorded_width),buffer in zip(module.OPERANDS,p.operand_offsets,p.operand_buffers):
                    self.assertEqual((recorded_name,offset,recorded_width),(name,cursor,width))
                    self.assertEqual((len(buffer.values[row]),buffer.values[row][0],buffer.values[row][-1]),(width,base+cursor,base+cursor+width-1));cursor+=width
            p.close();self.assertEqual(p.operand_buffers,[]);self.assertEqual(p.boundary_buffers,[])
            with self.assertRaises(ValueError):p.run([object()]*5)

    def test_original_profile_guard_preserved(self):
        module,_,_=isolated()
        for b,c,t in ((2,4096,4),(8,4096,4),(4,512,4),(4,4096,32),(True,4096,4)):
            with self.assertRaises(ValueError):module.Decode35EndpointProgram('checkpoint',{},IO(),b,c,t)

    def test_binding_restores_and_prefill_unchanged(self):
        module,_,base=isolated();parent=types.ModuleType('xModel.qwen35.vl_0_8b');decoder=types.ModuleType(parent.__name__+'.cuda_packed_decoder');parent.cuda_packed_decoder=decoder;decoder.CUDAPackedDecoderProgram=base
        def factory():
            self.assertIsInstance(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,4),module.Decode35EndpointProgram)
            self.assertIsInstance(decoder.CUDAPackedDecoderProgram('checkpoint',{},IO(),4,4096,32),base)
            raise RuntimeError('intentional')
        with patch.dict(sys.modules,{parent.__name__:parent,decoder.__name__:decoder}):
            with self.assertRaisesRegex(RuntimeError,'intentional'):module.construct_with_decode35_endpoint_program(factory)
        self.assertIs(decoder.CUDAPackedDecoderProgram,base)

    def test_native_seven_stage_function_and_geometry_unchanged(self):
        module,old,_=isolated();self.assertIs(module.Qwen35Decode35CUDAChunk,old.Qwen35Decode35CUDAChunk)
        self.assertEqual(module.NATIVE_OPERANDS,old.OPERANDS);self.assertEqual(len(module.NATIVE_OPERANDS),7)
        self.assertEqual(sum(w for _,w in module.NATIVE_OPERANDS),106496)
        self.assertEqual([n for n,_ in module.OPERANDS],['masked_input','input_normalized','projected_Q_gate','projected_K','projected_V','K_post_RoPE'])
        self.assertEqual(sum(w for _,w in module.OPERANDS),102400)

    def test_actual_forward_views_skip_only_pre_rope_and_preserve_native_key_offset(self):
        module,_,_=isolated();tree=ast.parse((M/'decode35_endpoint_program.py').read_bytes())
        forward=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward')
        block=next(n for n in ast.walk(forward) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=="layer['index'] == 35")
        code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),str(M/'decode35_endpoint_program.py'),'exec')
        for batch in (1,4):
            views=[];env={'layer':{'index':35,'kind':'full_attention','payload_width':0},'t':4,'h':5120,'w1':17,'w2':19,'b':batch,'packed':object(),'NATIVE_OPERANDS':module.NATIVE_OPERANDS,'operand_outputs':views,'packed_view':lambda packed,start,width,shape:(start,width,shape)}
            exec(code,env);cursor=4*5120+17+19;expected=[]
            for name,width in module.NATIVE_OPERANDS:
                if name!='K_normalized_pre_RoPE':expected.append((cursor,width,[batch,width]))
                cursor+=width
            self.assertEqual(views,expected);self.assertEqual(env['cursor'],cursor);self.assertEqual(len(views),6)
            self.assertEqual(views[-1][0],4*5120+17+19+102400)

    def test_exact_derivation_changes_only_observation_selection(self):
        d=json.loads((M/'evidence/decode35-endpoint-program-derivation.json').read_bytes());old=(M/d['source']).read_bytes();new=(M/d['target']).read_bytes()
        self.assertEqual(hashlib.sha256(old).hexdigest(),d['source_sha256']);self.assertEqual(hashlib.sha256(new).hexdigest(),d['target_sha256'])
        text=new.decode();self.assertTrue(text.startswith(d['header']));body=text[len(d['header']):]
        for c in reversed(d['class_changes']):self.assertEqual(body.count(c['new']),c['count']);body=body.replace(c['new'],c['old'])
        self.assertEqual(body,old.decode()[old.decode().index('class Decode35OperandProgram'):])
        self.assertFalse(d['new_native_guard_or_math_or_DLL_change'])

if __name__=='__main__':unittest.main()
