from pathlib import Path
import copy,hashlib,importlib.util,json,struct,sys,types,unittest,zipfile
from unittest.mock import patch
from .full_chunk_stage_layout import STAGES,stage_offsets,check_graphs
from .full_chunk_stage_replay import array
from .test_layer_boundary import Buffer,IO
M=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()
class Base:
    def __init__(self,*args,**kwargs):self.io=args[2];self.graph=True;self.size=args[5];self.output_width=17;self.mutate=False
    def run(self,inputs):
        if self.mutate:inputs[0]='mutated'
        return range(self.output_width)
    def close(self):self.graph=None
def study():
    name='xModel.qwen35.vl_0_8b.cuda_packed_program';base=types.ModuleType(name);base.CUDAPackedDecoderProgram=Base
    for k in ('T','flat_concat','concat','packed_view'):setattr(base,k,None)
    chunk=types.ModuleType('test2026.cuda.qwen38_5090.full_chunk_stage_chunk');chunk.FullChunkStageChunk=None
    spec=importlib.util.spec_from_file_location('test2026.cuda.qwen38_5090._isolated_full_chunk_stage_program',M/'full_chunk_stage_program.py');module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{name:base,chunk.__name__:chunk}):spec.loader.exec_module(module)
    return module
class FullChunkStageObservationTests(unittest.TestCase):
    def test_private_chunk_and_forward_are_exact_additive_observations(self):
        p=json.loads((M/'evidence/full-chunk-stage-program-derivation.json').read_bytes())
        with zipfile.ZipFile(M/'native_full_chunk_stage_inputs/qualified-kv128-source.zip') as z:
            for row in p['derivations']:
                raw=z.read(row['source']);self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode();target=(M/row['target']).read_bytes();self.assertEqual(sha(target),row['target_sha256'])
                if 'body_steps' in row:
                    old=text[text.index('@T.fusion('):];new=target.decode()[target.decode().index('@T.fusion('):];steps=row['body_steps']
                else:
                    old=text[text.index('    def _forward(self):'):text.index('    def run(self,inputs):')];new=target.decode()[target.decode().index('    def _forward(self):'):target.decode().index('    def input_witness(self,inputs,start):')];steps=row['method_steps']
                for c in reversed(steps):self.assertEqual(new.count(c['new']),1);new=new.replace(c['new'],c['old'])
                self.assertEqual(new,old)
        self.assertEqual(sum(w for n,w in STAGES),51200)
        offsets=stage_offsets(128,17);self.assertEqual(offsets[2],('qg',17+128*10240,128*12288));self.assertEqual(offsets[-1][1]+offsets[-1][2],17+128*51200)
        for name in ('full-chunk-stage-probe-derivation.json','full-chunk-stage-batch-derivation.json'):
            record=json.loads((M/'evidence'/name).read_bytes());raw=(M/record['source']).read_bytes();self.assertEqual(sha(raw),record['source_sha256'])
            for step in record['steps']:self.assertEqual(raw.count(step['old'].encode()),step['count']);raw=raw.replace(step['old'].encode(),step['new'].encode())
            self.assertEqual(raw,(M/record['target']).read_bytes());self.assertEqual(sha(raw),record['target_sha256'])
        from .test_kv128_prefix_cache import Prefix
        for tokens in (32,128):
            for enabled in (False,True):
                graph=Prefix().graphs(tokens)
                for key,value in graph.items():value['native_full_chunk_stage_witnesses']=int(enabled and key==str(tokens))
                check_graphs(graph,tokens,enabled);graph[str(tokens)]['native_full_chunk_stage_witnesses']=True
                with self.assertRaises(AssertionError):check_graphs(graph,tokens,enabled)

    def test_primary_prefix_all_stage_offsets_and_input_mutation_rejection(self):
        s=study();p=s.FullChunkStageProgram('checkpoint',{},IO(),1,4096,32);p.input_witness=lambda inputs,start:dict(start=start,values=tuple(inputs));inputs=list(range(5));out=p.run(inputs)
        self.assertEqual(out.shape,[1,17]);self.assertEqual(out.values,range(17));self.assertEqual(p.output_width,17+32*51200)
        for name,offset,words in stage_offsets(32,17):self.assertEqual(p.stage_buffers[name].values,range(offset,offset+words))
        self.assertEqual(p.readonly_observations[0]['before'],p.readonly_observations[0]['after']);self.assertEqual(p.stage_runs,1)
        p.mutate=True
        with self.assertRaises(AssertionError):p.run(inputs)
        with self.assertRaises(ValueError):p.run(inputs[:4])
        p.close()
        with self.assertRaises(ValueError):p.run(inputs)
        for batch,cap,t in ((2,4096,32),(1,512,32),(1,4096,4)):
            with self.assertRaises(ValueError):s.FullChunkStageProgram('checkpoint',{},IO(),batch,cap,t)

    def test_private_constructor_binding_restores_on_error_and_T4_unchanged(self):
        s=study();parent=types.ModuleType('xModel.qwen35.vl_0_8b');module=types.ModuleType(parent.__name__+'.cuda_packed_decoder');module.CUDAPackedDecoderProgram=Base;parent.cuda_packed_decoder=module
        def factory():
            self.assertIs(type(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),1,4096,4)),Base)
            self.assertIsInstance(module.CUDAPackedDecoderProgram('checkpoint',{},IO(),1,4096,128),s.FullChunkStageProgram)
            raise RuntimeError('retained synthetic error')
        with patch.dict(sys.modules,{parent.__name__:parent,module.__name__:module}):
            with self.assertRaises(RuntimeError):s.construct_with_full_chunk_stages(factory)
        self.assertIs(module.CUDAPackedDecoderProgram,Base)

    def test_QG_complete_bounded_array_corruption_nonfinite_and_repeat_faults_rejected(self):
        words=128*12288;files={'a.f32':bytes(1048576*4),'b.f32':bytes((words-1048576)*4)};raw=b''.join(files.values());entry=dict(words=words,sha256=sha(raw),repeat_sha256=sha(raw),parts=[dict(file=n,offset=i*1048576,words=len(b)//4,sha256=sha(b)) for i,(n,b) in enumerate(files.items())]);self.assertEqual(array(entry,files.__getitem__),raw)
        for field,value in (('repeat_sha256','f'*64),('words',True)):
            bad=copy.deepcopy(entry);bad[field]=value
            with self.assertRaises(AssertionError):array(bad,files.__getitem__)
        for field,value in (('file','../a.f32'),('offset',1),('words',True)):
            bad=copy.deepcopy(entry);bad['parts'][0][field]=value
            with self.assertRaises(AssertionError):array(bad,files.__getitem__)
        files['a.f32']=b'x'+files['a.f32'][1:]
        with self.assertRaises(AssertionError):array(entry,files.__getitem__)
        b=bytearray(files['a.f32']);struct.pack_into('<f',b,0,float('inf'));files['a.f32']=bytes(b);entry['parts'][0]['sha256']=sha(files['a.f32']);entry['sha256']=entry['repeat_sha256']=sha(b''.join(files.values()))
        with self.assertRaises(AssertionError):array(entry,files.__getitem__)
