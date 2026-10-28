"""Check call identity and the actual speculative prefill binding protocol."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from .native_stage_timing import StageRecorder
from .native_stage_attachment import StageAttachment,PhaseCall

class Program:
    def run(self,*args,**kwargs):return (args,kwargs)

class Verifier:
    pending=False
    def verify(self,*args,**kwargs):return (args,kwargs)
    def commit_prefix(self,*args,**kwargs):return (args,kwargs)
    def _read(self,*args,**kwargs):return (args,kwargs)

class Decoder:
    def __init__(self):
        self.programs={4:Program(),128:Program()};self.slots=[8];self.last_hidden=object()
        self.head=SimpleNamespace(run_device=lambda values:values)
    def chunk_hidden(self,*args,**kwargs):return self.programs[128].run(*args,**kwargs)
    def prefill_sequence(self,hidden,positions,lengths):
        self.chunk_hidden(hidden,positions,lengths);return [123]

def actual_prefill():
    path=Path(__file__).resolve().parents[3]/'xModel/qwen35/vl_0_8b/cuda_speculative.py'
    tree=ast.parse(path.read_bytes());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CUDASpeculativeSession')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prefill')
    scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope);return scope['prefill']

class Session:
    prefill=actual_prefill()
    def __init__(self):
        self.decoder=Decoder();self.verifier=Verifier();self.program=Program();self.prefill16_program=Program()
        self.reset_required=False;self.poisoned=False;self.ready=False;self.head_candidates=0;self.slots=[8];self.batch=1
    def _prefill_hook(self,*args,**kwargs):return self._original_chunk(*args,**kwargs)
    def _step(self,*args,**kwargs):return (args,kwargs)
    def generate(self,*args,**kwargs):return (args,kwargs)
    def reset(self,*args,**kwargs):return (args,kwargs)
    def _draft(self,*args,**kwargs):return (args,kwargs)
    def _correct(self,*args,**kwargs):return (args,kwargs)
    def _call(self,*args,**kwargs):return (args,kwargs)

class AttachmentControls(unittest.TestCase):
    def fixture(self):
        events=[];ticks=iter(range(10000));rec=StageRecorder(events.append,lambda:next(ticks));session=Session()
        return session,rec,events
    def test_startup_is_quiet_and_values_kwargs_and_call_order_unchanged(self):
        s,r,events=self.fixture();a=StageAttachment(s,r);token=object()
        result=s.generate(token,flag=token);self.assertIs(result[0][0],token);self.assertIs(result[1]['flag'],token);self.assertEqual(events,[])
        r.start_request('a'*32);result=s.generate(token,flag=token);r.finish_request()
        self.assertIs(result[0][0],token);self.assertIs(result[1]['flag'],token)
        self.assertEqual([(e['stage'],e['phase']) for e in events],[('worker_request','begin'),('session_generate','begin'),('session_generate','end'),('worker_request','end')]);a.close();a.close()
    def test_actual_prefill_temporary_override_restores_attached_target_method(self):
        s,r,events=self.fixture();self.assertNotIn('chunk_hidden',s.decoder.__dict__);a=StageAttachment(s,r);wrapped=s.decoder.chunk_hidden
        r.start_request('b'*32);self.assertEqual(s.prefill([object()],[[[0,0,0]]],[1],[1]),[123]);r.finish_request()
        self.assertIs(s.decoder.chunk_hidden,wrapped);self.assertTrue(s.ready)
        self.assertEqual([e['stage'] for e in events if e['phase']=='begin'],['worker_request','session_prefill','target_chunk','target_program_t128'])
        a.close();self.assertNotIn('chunk_hidden',s.decoder.__dict__);self.assertEqual(s.decoder.chunk_hidden.__func__,Decoder.chunk_hidden)
    def test_actual_prefill_exception_identity_and_binding_restoration(self):
        s,r,events=self.fixture();error=RuntimeError('original target error')
        def fail(*args,**kwargs):raise error
        s.decoder.chunk_hidden=fail;a=StageAttachment(s,r);wrapped=s.decoder.chunk_hidden;r.start_request('c'*32)
        with self.assertRaises(RuntimeError) as caught:s.prefill([],[],[],[])
        self.assertIs(caught.exception,error);self.assertIs(s.decoder.chunk_hidden,wrapped);self.assertTrue(s.poisoned)
        r.finish_request(False);self.assertTrue(all(e['call_succeeded'] is False for e in events if e['phase']=='end'))
        a.close();self.assertIs(s.decoder.chunk_hidden,fail)
    def test_restore_instance_override_and_preserve_independent_later_change(self):
        s,r,_=self.fixture();original=lambda *args:None;s.reset=original;a=StageAttachment(s,r);later=lambda *args:None;s.generate=later
        with self.assertRaisesRegex(ValueError,'generate'):a.close()
        self.assertIs(s.reset,original);self.assertIs(s.generate,later);self.assertNotIn('chunk_hidden',s.decoder.__dict__)
    def test_missing_boundary_or_active_request_does_not_partially_patch(self):
        s,r,_=self.fixture();s.prefill16_program=None
        with self.assertRaises(AttributeError):StageAttachment(s,r)
        self.assertNotIn('prefill',s.__dict__);self.assertNotIn('chunk_hidden',s.decoder.__dict__)
        s,r,_=self.fixture();r.start_request('d'*32)
        with self.assertRaises(ValueError):StageAttachment(s,r)
        r.finish_request();self.assertNotIn('prefill',s.__dict__)
    def test_phase_labels_keep_arguments_and_never_turn_host_time_into_kernel_time(self):
        s,r,events=self.fixture();wrapped=PhaseCall(s._call,r);value=object();r.start_request('e'*32)
        self.assertIs(wrapped(value,value,False,'prefill')[0][0],value)
        self.assertIs(wrapped(value,stage='correction')[0][0],value);r.finish_request()
        self.assertEqual([e['stage'] for e in events if e['phase']=='begin'],['worker_request','mtp_prefill','mtp_correction'])
        self.assertTrue(all(e['clock_domain']=='native_worker' and e['CUDA_kernel_elapsed_claimed'] is False for e in events))

if __name__=='__main__':unittest.main()
