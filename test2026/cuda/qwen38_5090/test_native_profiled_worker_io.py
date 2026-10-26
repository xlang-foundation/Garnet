"""CPU-only exact frame/Unicode/partial-write/error and model-operation controls."""
from pathlib import Path
from unittest.mock import patch
import ast,io,json,unittest
from . import native_profiled_worker_io as profiled
from . import native_worker_io as original

M=Path(__file__).resolve().parent

class Controls(unittest.TestCase):
    def expected(self,value):
        result=bytearray()
        def write(raw):result.extend(raw);return len(raw)
        original.emit(value,write=write);return bytes(result)

    def test_exact_original_ascii_unicode_frames_and_ids(self):
        for value in ({'event':'ready','batch':1},{'event':'token','text':'中文 🚀','token_id':123,'sampled_tokens':1},
                      {'event':'completed','text':'snow 雪','generated_ids':[1,2,248046]}):
            stream=io.StringIO()
            with patch.object(profiled.sys,'stdout',stream),patch.object(original.os,'write',side_effect=AssertionError('Must not use invalid raw descriptor')):
                profiled.emit(value)
            self.assertEqual(stream.getvalue().encode('ascii'),self.expected(value))

    def test_partial_ascii_writes_keep_every_byte_and_flush(self):
        class ShortWriter:
            def __init__(self):self.data='';self.flushes=0
            def write(self,text):n=min(7,len(text));self.data+=text[:n];return n
            def flush(self):self.flushes+=1
        stream=ShortWriter();value={'event':'token','text':'a\nb 雪','token_id':42}
        with patch.object(profiled.sys,'stdout',stream):profiled.emit(value)
        self.assertEqual(stream.data.encode('ascii'),self.expected(value));self.assertGreater(stream.flushes,1)

    def test_errors_propagate_without_silent_fallback(self):
        error=OSError('explicit synthetic write failure')
        class Broken:
            def write(self,text):raise error
            def flush(self):raise AssertionError('No flush after failed write')
        with patch.object(profiled.sys,'stdout',Broken()),self.assertRaises(OSError) as caught:profiled.emit({'event':'ready'})
        self.assertIs(caught.exception,error)
        class Empty:
            def write(self,text):return 0
            def flush(self):pass
        with patch.object(profiled.sys,'stdout',Empty()),self.assertRaises(OSError):profiled.emit({'event':'ready'})

    def test_original_input_parser_and_bounds_remain_shared(self):
        self.assertIs(profiled.requests,original.requests)
        rows=iter([b'{"id":"one","body":',b'{}}\n',b''])
        self.assertEqual(list(profiled.requests(read=lambda:next(rows))),[{'id':'one','body':{}}])
        for raw in (b'{"id":1}',b'x'*(original.MAX_FRAME_BYTES+1)):
            data=iter([raw,b''])
            with self.assertRaises(ValueError):list(profiled.requests(read=lambda:next(data)))

    def test_model_worker_changes_only_diagnostic_transport_import(self):
        candidate=ast.parse((M/'native_cuda_phase_stdio_worker.py').read_bytes());original_tree=ast.parse((M/'native_cuda_phase_logged_worker.py').read_bytes())
        changed=0
        for node in candidate.body:
            if isinstance(node,ast.ImportFrom) and node.module=='test2026.cuda.qwen38_5090.native_profiled_worker_io':
                node.module='test2026.cuda.qwen38_5090.native_worker_io';changed+=1
        self.assertEqual(changed,1);self.assertEqual(ast.dump(candidate,include_attributes=False),ast.dump(original_tree,include_attributes=False))

    def test_failed_parent_requires_recorded_ready_stdout_error_and_cleanup(self):
        import copy
        from .prepare_native_cuda_stdio_profiled import admit_parent
        baseline='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
        q={'status':'native_cuda_entry_profiled_failure_preserved','archive_sha256':'a'*64,'profiled':True,
           'performance_admitted':False,'two_times_target_achieved':False,'library_sha256':baseline,'control_replay':None,
           'diagnostic_logs':{'native/control/host-stages.entry.jsonl':[{'stage':'worker_raised','exception':
               {'type':'OSError','repr':"OSError('write failed')",'traceback':'os.write(1,raw)'}}],
               'native/control/host-stages.bootstrap.jsonl':[{'stage':'native_ready_observed_before_pipe_emit'}]}}
        w={'exit_code':1,'cleanup_finished':True,'library_unchanged':True,'library_sha256':baseline,
           'source_sha256':{'s':'b'},'source_after_sha256':{'s':'b'},
           'runtime_dependencies':{'garnet.dll':baseline},'runtime_dependencies_after':{'garnet.dll':baseline}}
        admit_parent(q,w,'a'*64)
        for fault in ('exit0','bool_exit','cleanup','source','error','ready','requests','promoted'):
            p,a=copy.deepcopy(q),copy.deepcopy(w)
            if fault=='exit0':a['exit_code']=0
            elif fault=='bool_exit':a['exit_code']=True
            elif fault=='cleanup':a['cleanup_finished']=False
            elif fault=='source':a['source_after_sha256']['s']='changed'
            elif fault=='error':p['diagnostic_logs']['native/control/host-stages.entry.jsonl'][0]['exception']['type']='ValueError'
            elif fault=='ready':p['diagnostic_logs']['native/control/host-stages.bootstrap.jsonl']=[]
            elif fault=='requests':p['control_replay']={'unexpected_request':True}
            else:p['performance_admitted']=True
            with self.subTest(fault=fault),self.assertRaises(AssertionError):admit_parent(p,a,'a'*64)

    def test_preloaded_control_still_requires_all130_events_and_exact_ids(self):
        import copy
        from .probe_native_cuda_stdio_bootstrap import validate_preloaded_events
        key='synthetic-stdio-only';case={'prompt_ids':[1]*3968,'generated_ids_by_mode':{'ordinary':[3]*128}}
        events=[{'event':'ready','generation_mode':'ordinary','batch':1,'capacity':4096,'library_sha256':'synthetic-library'}]
        events.extend({'event':'token','id':key,'sampled_tokens':i+1,'token_id':3,'prompt_tokens':3968} for i in range(128))
        events.append({'event':'completed','id':key,'prompt_ids':[1]*3968,'generated_ids':[3]*128,
                       'image_sha256':None,'finish_reason':'length','ignore_eos':True})
        validate_preloaded_events(events,case,key,'synthetic-library')
        for fault in ('missing','id','ordinal','bool_batch','completion','prompt','stop'):
            bad=copy.deepcopy(events)
            if fault=='missing':bad.pop(5)
            elif fault=='id':bad[4]['token_id']=4
            elif fault=='ordinal':bad[4]['sampled_tokens']=3
            elif fault=='bool_batch':bad[0]['batch']=True
            elif fault=='completion':bad[-1]['generated_ids'][-1]=4
            elif fault=='prompt':bad[-1]['prompt_ids'][-1]=2
            else:bad[-1]['finish_reason']='stop'
            with self.subTest(fault=fault),self.assertRaises(AssertionError):validate_preloaded_events(bad,case,key,'synthetic-library')

if __name__=='__main__':unittest.main()
