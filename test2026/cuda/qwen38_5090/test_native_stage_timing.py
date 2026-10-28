import unittest
from .native_stage_timing import StageRecorder,TimedBridgeLock


class Timing(unittest.TestCase):
    def recorder(self):
        events=[];counter=iter(range(1000));return StageRecorder(events.append,lambda:next(counter)),events

    def test_nested_calls_keep_values_identity_order_and_declared_host_clock(self):
        recorder,events=self.recorder();value=object();calls=[]
        inner=recorder.wrap(lambda x: calls.append('inner') or x,'draft_prefill')
        outer=recorder.wrap(lambda x: inner(x),'prefill_hook')
        recorder.start_request('a'*32);self.assertIs(outer(value),value);recorder.finish_request()
        self.assertEqual(calls,['inner']);self.assertEqual([e['sequence'] for e in events],list(range(1,7)))
        self.assertEqual([(e['stage'],e['phase']) for e in events],[('worker_request','begin'),('prefill_hook','begin'),('draft_prefill','begin'),('draft_prefill','end'),('prefill_hook','end'),('worker_request','end')])
        self.assertEqual([e['parent_span'] for e in events],[None,1,2,2,1,None]);self.assertTrue(all(e['clock_domain']=='native_worker' and e['CUDA_kernel_elapsed_claimed'] is False for e in events))

    def test_startup_calls_are_uninstrumented_and_kwargs_unchanged(self):
        recorder,events=self.recorder();call=recorder.wrap(lambda *,x:x,'target_prefill');value=object();self.assertIs(call(x=value),value);self.assertEqual(events,[])

    def test_exception_identity_and_failure_span_are_preserved(self):
        recorder,events=self.recorder();error=ValueError('original')
        def fail():raise error
        recorder.start_request('b'*32)
        with self.assertRaises(ValueError) as caught:recorder.wrap(fail,'target_prefill')()
        self.assertIs(caught.exception,error);self.assertFalse(events[-1]['call_succeeded']);recorder.finish_request(False);self.assertFalse(events[-1]['call_succeeded']);self.assertIsNone(recorder.active)

    def test_request_reentry_unbalanced_spans_bad_labels_and_cross_request_close_rejected(self):
        recorder,_=self.recorder()
        for key in ('short','g'*32,123):
            with self.assertRaises(ValueError):recorder.start_request(key)
        with self.assertRaises(ValueError):recorder.begin('target_prefill')
        root=recorder.start_request('c'*32)
        with self.assertRaises(ValueError):recorder.start_request('d'*32)
        with self.assertRaises(ValueError):recorder.begin('arbitrary input text')
        child=recorder.begin('target_prefill')
        with self.assertRaises(ValueError):recorder.end(root)
        with self.assertRaises(ValueError):recorder.finish_request()
        recorder.end(child);recorder.finish_request()
        with self.assertRaises(ValueError):recorder.end(child)
        recorder.start_request('d'*32);another=recorder.begin('target_prefill')
        with self.assertRaises(ValueError):recorder.end(child)
        recorder.end(another);recorder.finish_request()

    def test_bridge_lock_releases_and_preserves_original_exception(self):
        events=[];calls=[]
        class Lock:
            def acquire(self):calls.append('acquire')
            def release(self):calls.append('release')
        clock=iter((10,13,20));lock=TimedBridgeLock(Lock(),lambda:'e'*32,events.append,lambda:next(clock))
        error=RuntimeError('worker failure')
        with self.assertRaises(RuntimeError) as caught:
            with lock:raise error
        self.assertIs(caught.exception,error);self.assertEqual(calls,['acquire','release']);self.assertEqual([e['phase'] for e in events],['queued','acquired','released']);self.assertEqual([e['bridge_monotonic_seconds'] for e in events],[10,13,20]);self.assertTrue(all(e['clock_domain']=='http_bridge' for e in events));self.assertFalse(events[-1]['call_succeeded'])

    def test_logging_failure_after_acquire_or_at_release_never_strands_lock(self):
        for failure_phase in ('acquired','released'):
            calls=[]
            class Lock:
                def acquire(self):calls.append('acquire')
                def release(self):calls.append('release')
            def emit(event):
                if event['phase']==failure_phase:raise OSError('log failed')
            lock=TimedBridgeLock(Lock(),lambda:'f'*32,emit)
            with self.assertRaises(OSError):
                with lock:pass
            self.assertEqual(calls,['acquire','release'])

if __name__=='__main__':unittest.main()
