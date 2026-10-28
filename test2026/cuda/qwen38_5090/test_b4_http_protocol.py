"""Pure CPU checks for proposed HTTP validation, grouping and frame bounds."""
from pathlib import Path
import ast,copy,importlib.util,json,unittest
from .b4_serving_request import validate_body,partition_jobs
M=Path(__file__).resolve().parent;D=M/'native_b4_http_protocol_candidate'

def functions(path):
    return {n.name:n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}

def bridge_validator():
    node=functions(D/'native_http_bridge.py')['validate_body'];scope={}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(D/'native_http_bridge.py'),'exec'),scope)
    return scope['validate_body']

def load_io():
    spec=importlib.util.spec_from_file_location('isolated_b4_transport_candidate',D/'native_worker_io.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def body(limit=128,ignore=False):
    return dict(model='qwen38-27b-nvfp4',temperature=0,chat_template_kwargs=dict(enable_thinking=False),max_tokens=limit,ignore_eos=ignore,messages=[dict(role='user',content='test')])

class HTTPProtocolControls(unittest.TestCase):
    def test_body_policy_matches_qualified_native_validator_and_HTTP_candidate(self):
        native=functions(M/'native_o128_stream_worker.py')['validate_body'];local=functions(M/'b4_serving_request.py')['validate_body'];http=functions(D/'native_http_bridge.py')['validate_body']
        self.assertEqual(ast.dump(native),ast.dump(local));self.assertEqual(ast.dump(native),ast.dump(http))
        for call in (validate_body,bridge_validator()):
            for ignore in (False,True):self.assertIsNone(call(body(ignore=ignore)))
            for flag in (None,0,1,'false'):
                b=body();b['ignore_eos']=flag
                with self.assertRaises(ValueError):call(b)
    def test_grouping_keeps_all_real_IDs_bodies_and_stable_homogeneous_groups(self):
        jobs=[dict(id=hex(i+1)[2:].zfill(32),body=body(limit,ignore)) for i,(limit,ignore) in enumerate(((128,False),(8,False),(128,True),(128,False)))]
        before=copy.deepcopy(jobs);groups=partition_jobs(jobs)
        self.assertEqual([[j['id'] for j in group] for group in groups],[[jobs[0]['id'],jobs[3]['id']],[jobs[1]['id']],[jobs[2]['id']]])
        self.assertEqual(jobs,before);self.assertEqual(sorted(j['id'] for g in groups for j in g),sorted(j['id'] for j in jobs))
    def test_invalid_frame_or_late_invalid_job_is_rejected(self):
        jobs=[dict(id=hex(i+1)[2:].zfill(32),body=body()) for i in range(4)]
        for change in ('empty','too_many','duplicate','bad_id','late_flag'):
            trial=copy.deepcopy(jobs)
            if change=='empty':trial=[]
            elif change=='too_many':trial+=copy.deepcopy(trial[:1])
            elif change=='duplicate':trial[-1]['id']=trial[0]['id']
            elif change=='bad_id':trial[-1]['id']='short'
            else:trial[-1]['body']['ignore_eos']=1
            with self.assertRaises(ValueError):partition_jobs(trial)
    def test_existing_model_template_limit_and_image_bounds_are_preserved(self):
        for change in ('model','temperature','thinking','limit','messages','image'):
            b=body()
            if change=='model':b['model']='different'
            elif change=='temperature':b['temperature']=0.5
            elif change=='thinking':b['chat_template_kwargs']['enable_thinking']=True
            elif change=='limit':b['max_tokens']=True
            elif change=='messages':b['messages']=[]
            else:b['messages'][0]['content']=[dict(type='text',text='wrong order'),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,/9j/'))]
            for call in (validate_body,bridge_validator()):
                with self.assertRaises(ValueError):call(b)
    def test_default_2MiB_bound_and_explicit_B4_bound(self):
        io=load_io();wire=(json.dumps(dict(payload='x'*(2*1024*1024)))+'\n').encode()
        with self.assertRaises(ValueError):list(io.requests(iter((wire,b'')).__next__))
        got=list(io.requests(iter((wire,b'')).__next__,max_frame_bytes=16*1024*1024));self.assertEqual(len(got[0]['payload']),2*1024*1024)
    def test_fragmented_frames_and_incomplete_EOF(self):
        io=load_io();blocks=iter((b'{"a":',b'1}\n{"b":"',b'two"}\n',b''))
        self.assertEqual(list(io.requests(blocks.__next__,max_frame_bytes=100)),[dict(a=1),dict(b='two')])
        with self.assertRaises(ValueError):list(io.requests(iter((b'{"a":1}',b'')).__next__))
    def test_invalid_frame_bound_rejected_before_reading(self):
        io=load_io()
        def never_read():self.fail('Invalid bound consumed input')
        for value in (True,0,-1,None,16.0,32*1024*1024+1):
            with self.subTest(value=value),self.assertRaises(ValueError):list(io.requests(never_read,max_frame_bytes=value))
    def test_only_parser_and_HTTP_validation_functions_change(self):
        for filename,changed in (('native_worker_io.py','requests'),('native_http_bridge.py','validate_body')):
            before=functions(M/filename);after=functions(D/filename);self.assertEqual(set(before),set(after))
            for name in before:
                if name!=changed:self.assertEqual(ast.dump(before[name]),ast.dump(after[name]),name)
        self.assertEqual(load_io().MAX_FRAME_BYTES,2*1024*1024)

if __name__=='__main__':unittest.main()
