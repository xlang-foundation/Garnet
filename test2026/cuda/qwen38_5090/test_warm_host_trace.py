"""CPU-only integrity tests for transparent nested diagnostic intervals."""
import unittest
from warm_host_trace import WarmHostTrace


class WarmHostTraceTests(unittest.TestCase):
    def test_transparency_parent_accounting_and_restore(self):
        class Subject:
            def inner(self,value):return value+1
            def outer(self,value):return self.inner(value)*2
        original=Subject.inner;clock=iter((0.,1.,3.,5.))
        trace=WarmHostTrace(clock=lambda:next(clock))
        trace.instrument(Subject,'outer','outer');trace.instrument(Subject,'inner','inner')
        try:
            self.assertEqual(Subject().outer(3),8);self.assertEqual(trace.records,[])
            trace.begin('warm-case');self.assertEqual(Subject().outer(3),8);trace.end()
            summaries={r['stage']:r for r in trace.summaries()}
            self.assertEqual(summaries['outer']['inclusive_host_wall_seconds'],5.)
            self.assertEqual(summaries['outer']['exclusive_host_wall_seconds'],3.)
            self.assertEqual(summaries['inner']['exclusive_host_wall_seconds'],2.)
            self.assertEqual(trace.records[0]['parent'],trace.records[1]['number'])
        finally:trace.restore()
        self.assertIs(Subject.inner,original)

    def test_original_failure_is_preserved_and_capture_can_close(self):
        class Subject:
            def fail(self):raise ValueError('original failure')
        trace=WarmHostTrace();trace.instrument(Subject,'fail','failure')
        try:
            trace.begin('failure')
            with self.assertRaisesRegex(ValueError,'original failure'):Subject().fail()
            trace.end();self.assertEqual(trace.summaries()[0]['errors'],1)
        finally:trace.restore()

    def test_rejects_unbalanced_lifecycle(self):
        trace=WarmHostTrace()
        with self.assertRaisesRegex(RuntimeError,'Unbalanced'):trace.end()
        trace.begin('active')
        with self.assertRaisesRegex(RuntimeError,'Nested'):trace.begin('nested')
        with self.assertRaisesRegex(RuntimeError,'active'):trace.restore()
        trace.end();trace.restore()

    def test_inherited_method_restores_without_creating_subclass_override(self):
        class Parent:
            def call(self):return 'unchanged'
        class Child(Parent):pass
        trace=WarmHostTrace();trace.instrument(Child,'call','inherited')
        trace.begin('child');self.assertEqual(Child().call(),'unchanged');trace.end();trace.restore()
        self.assertNotIn('call',Child.__dict__);self.assertIs(Child.call,Parent.call)


if __name__=='__main__':unittest.main()
