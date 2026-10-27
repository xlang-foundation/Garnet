"""Execute actual orchestration/worker code with token stubs; no native imports or ML replay."""
import ast,hashlib,json,os,sys,tempfile,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch

R=Path(__file__).resolve().parents[3]
M=Path(__file__).resolve().parent
tree=ast.parse((R/'runtime/cuda_serving.py').read_bytes())
model=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Model')
methods=[n for n in model.body if isinstance(n,ast.FunctionDef) and n.name in ('infer_batch','infer')]
assert len(methods)==2
scope={'time':time,'hashlib':hashlib,'__file__':str(R/'runtime/cuda_serving.py')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='ActualMethods',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),str(R/'runtime/cuda_serving.py'),'exec'),scope)

class Tokenizer:
    def open(self,*args):pass
    def token_id(self,text):return 99 if text=='<|im_end|>' else 100
    def encode(self,text):return [1]
    def decode(self,tokens):return '|'.join(str(v) for v in tokens)

class Vision:
    def __init__(self,*args):self.graph=types.SimpleNamespace(enable_cuda_graph=lambda value:None)
    def close(self):pass

class Decoder:
    def __init__(self,sequence):
        self.sequence=sequence;self.steps=[];self.resets=0;self.closed=False
        self.io=types.SimpleNamespace(pack_prompt=lambda *args:[[0]],embedding=lambda ids:ids,row=lambda rows,i:rows[i])
    def reset(self):self.resets+=1;self.cursor=0
    def enable_cuda_graph(self,value):pass
    def prefill_sequence(self,*args):self.cursor=0;return list(self.sequence[0])
    def step(self,prediction,active=None,positions=None):
        self.steps.append((list(prediction),None if active is None else list(active),positions));self.cursor+=1
        return list(self.sequence[min(self.cursor,len(self.sequence)-1)])
    def memory_info(self):return {'graphs':[{'native_int4_g128_projection_weights':496}]}
    def close(self):self.closed=True

def instance(sequence):
    m=scope['ActualMethods']();m.batch=len(sequence[0]);m.max_output=128;m.lock=threading.RLock();m.speculative=None
    m.decoder=Decoder(sequence);m.tokenizer=Tokenizer();m.stops=[99,100];m.pack_prompt=True;m.name='test';m.profile={};m.vision=Vision()
    for n in ('prefill_chunk','decode_chunk','speculative_depth','speculative_candidates'):setattr(m,n,0)
    for n in ('shared_projection','packed_program','resident_controls','register_delta_128','flat_packed_concat','serial_target_graph_reuse','deferred_state_commit','native_prefix_commit','shared_output_storage','reuse_verified_head','serial_graph_reuse','speculative_replay'):setattr(m,n,False)
    m._prepare=lambda prompt,image:([0],[[0,0,0]],[1],1,0,0,None,-1)
    m.memory_info=lambda:{};m.close=m.decoder.close
    return m

class FixedOutputPolicyTests(unittest.TestCase):
    def test_default_stops_at_first_eos(self):
        m=instance([[99],[7]]);seen=[];r=m.infer_batch(['x'],[''],128,on_token=lambda *x:seen.append(x))
        self.assertEqual((r['generated_token_ids'],r['sampled_tokens_including_stop'],r['finish_reasons'],r['stop_token_ids']),([[]],[1],['eos'],[99]))
        self.assertEqual(len(seen),1);self.assertEqual(m.decoder.steps,[])
        self.assertEqual(r['generation_policy'],{'ignore_eos':False,'max_new_tokens':128})

    def test_ignore_eos_runs_all_128_and_keeps_raw_ids(self):
        m=instance([[99],[7],[99]]);seen=[];r=m.infer('x','',128,on_token=lambda *x:seen.append(x),ignore_eos=True)
        expected=[99,7]+[99]*126
        self.assertEqual(r['generated_token_ids'],[expected]);self.assertEqual([x[1] for x in seen],expected)
        self.assertEqual([x[2] for x in seen],list(range(1,129)));self.assertEqual(r['text'],'7')
        self.assertEqual((r['finish_reasons'],r['stop_token_ids'],r['sampled_tokens_including_stop'],r['decode_steps']),(['length'],[None],[128],127))

    def test_default_batch_masks_finished_rows(self):
        m=instance([[99,7],[8,99]]);seen=[];r=m.infer_batch(['x','y'],['',''],3,on_token=lambda *x:seen.append(x))
        self.assertEqual(r['generated_token_ids'],[[],[7]]);self.assertEqual(r['sampled_tokens_including_stop'],[1,2])
        self.assertEqual([x[:3] for x in seen],[(0,99,1),(1,7,1),(1,99,2)])
        self.assertEqual(m.decoder.steps[0][1],[0,1])

    def test_ignore_eos_keeps_all_batch_rows_active(self):
        m=instance([[99,7],[8,99],[99,8]]);r=m.infer_batch(['x','y'],['',''],3,ignore_eos=True)
        self.assertEqual(r['generated_token_ids'],[[99,8,99],[7,99,8]])
        self.assertEqual(r['texts'],['8','7|8']);self.assertEqual(r['sampled_tokens_including_stop'],[3,3])
        self.assertEqual([s[1] for s in m.decoder.steps],[[1,1],[1,1]])

    def test_nonboolean_rejected_before_reset(self):
        for flag in (1,0,'true',None):
            with self.subTest(flag=flag):
                m=instance([[7]])
                with self.assertRaisesRegex(TypeError,'boolean'):m.infer('x','',3,ignore_eos=flag)
                self.assertEqual(m.decoder.resets,0)

    def test_speculative_ignore_eos_rejected_before_reset(self):
        m=instance([[7]]);m.speculative=object()
        with self.assertRaisesRegex(ValueError,'target-only'):m.infer('x','',3,ignore_eos=True)
        self.assertEqual(m.decoder.resets,0)

    def test_original_positional_callback_supported(self):
        m=instance([[7],[8]]);seen=[];r=m.infer('x','',2,lambda *x:seen.append(x))
        self.assertEqual(r['generated_token_ids'],[[7,8]]);self.assertEqual(r['finish_reasons'],['length']);self.assertEqual(len(seen),2)

    def worker(self,name,flag='absent'):
        events=[];m=instance([[99],[7],[99]]);body={'model':'qwen38-27b-int4' if name=='native_int4_stream_worker.py' else 'qwen35-4b-bf16','temperature':0,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':'x'}],'max_tokens':128}
        if flag!='absent':body['ignore_eos']=flag
        def module(name,**items):
            value=types.ModuleType(name);value.__dict__.update(items);return value
        patches={'garnet':module('garnet',tokenizer=Tokenizer),
            'runtime.cuda_serving':module('runtime.cuda_serving',Model=lambda *args:m),
            'xModel.qwen38.vl_27b.model':module('xModel.qwen38.vl_27b.model',create_cuda_int4_packed_decoder=lambda *args:m.decoder,prepare_cuda_image=lambda *args:None),
            'xModel.qwen35.vl_0_8b.cuda_decoder':module('xModel.qwen35.vl_0_8b.cuda_decoder',CUDAImageEncoder=Vision),
            'test2026.cuda.qwen38_5090.native_worker_io':module('test2026.cuda.qwen38_5090.native_worker_io',requests=lambda:iter([{'id':'case','body':body}]),emit=lambda row:events.append(json.loads(json.dumps(row))))}
        with tempfile.TemporaryDirectory() as temp:
            Path(temp,'config.json').write_text(json.dumps({'vision_config':{}}));old_path=list(sys.path)
            try:
                with patch.dict(sys.modules,patches),patch.dict(os.environ,{'GARNET_RUNTIME_ROOT':temp}),patch.object(sys,'argv',[name,temp,str(Path(temp,'image.jpg'))]):
                    exec(compile((M/name).read_bytes(),str(M/name),'exec'),{'__name__':'__main__','__file__':str(M/name)})
            finally:sys.path[:]=old_path
        self.assertTrue(m.decoder.closed)
        return events,m

    def test_both_actual_workers_preserve_default_eos(self):
        for name in ('native_int4_stream_worker.py','native_4b_stream_worker.py'):
            with self.subTest(worker=name):
                events,_=self.worker(name);tokens=[e for e in events if e['event']=='token'];last=events[-1]
                self.assertEqual([e['token_id'] for e in tokens],[99]);self.assertTrue(tokens[0]['stop'])
                self.assertEqual(last['finish_reason'],'stop');self.assertEqual(last['generated_ids'],[99])

    def test_both_actual_workers_force_128_even_when_last_is_eos(self):
        for name in ('native_int4_stream_worker.py','native_4b_stream_worker.py'):
            with self.subTest(worker=name):
                events,m=self.worker(name,True);tokens=[e for e in events if e['event']=='token'];last=events[-1]
                self.assertEqual(len(tokens),128);self.assertEqual([e['sampled_tokens'] for e in tokens],list(range(1,129)))
                self.assertTrue(all(not e['stop'] for e in tokens));self.assertEqual(last['generated_ids'],[99,7]+[99]*126)
                self.assertEqual(last['text'],'7');self.assertEqual(last['finish_reason'],'length');self.assertEqual(len(m.decoder.steps),127)
                self.assertEqual(last['generation_policy'],{'ignore_eos':True,'max_new_tokens':128})

    def test_both_actual_workers_reject_nonboolean(self):
        for name in ('native_int4_stream_worker.py','native_4b_stream_worker.py'):
            for flag in (1,None,'true'):
                with self.subTest(worker=name,flag=flag),self.assertRaisesRegex(TypeError,'boolean'):self.worker(name,flag)

if __name__=='__main__':unittest.main()
