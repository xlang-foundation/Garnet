"""Exercise actual workspace policy and ownership, without loading native code."""
import unittest
from . import test_mtp_builder_workspace as workspace
from .test_mtp_prefill16_b4_study import decoder,load
from .test_mtp_prefill16_study import FakeBase

class PrefillWorkspaceControls(unittest.TestCase):
    def test_actual_program_allows_only_b4_t4_or_t16_at_256mib(self):
        helper=workspace.WorkspaceTests();cls,graphs=helper.load()
        for size in (4,16):
            obj=helper.make(cls,batch=4,capacity=4096,chunk_size=size,builder_workspace_bytes=256<<20)
            self.assertEqual(graphs[-1].workspace,[256<<20]);obj.close();self.assertTrue(graphs[-1].closed)
        count=len(graphs)
        for values in ((1,4096,16),(2,4096,16),(8,4096,16),(4,512,16),(4,8192,16),(4,4096,1),(4,4096,8),(4,4096,16.0),(4.0,4096,16),(4,4096.0,16),(True,4096,16)):
            with self.subTest(values=values),self.assertRaises(ValueError):helper.make(cls,batch=values[0],capacity=values[1],chunk_size=values[2],builder_workspace_bytes=256<<20)
        self.assertEqual(len(graphs),count)
    def test_program_setter_failure_closes_t16(self):
        helper=workspace.WorkspaceTests();cls,graphs=helper.load(fail=True)
        with self.assertRaisesRegex(RuntimeError,'setter failure'):helper.make(cls,batch=4,capacity=4096,chunk_size=16,builder_workspace_bytes=256<<20)
        self.assertTrue(graphs[0].closed)
    def test_default_and_explicit_prefill_policy_propagate_and_report(self):
        for options,expected in (({},128<<20),({'draft_prefill_builder_workspace_bytes':256<<20},256<<20)):
            obj=load()(decoder(),'fixture',[],**options)
            self.assertEqual(obj.prefill16_program.kwargs,dict(builder_workspace_bytes=expected))
            self.assertEqual(obj.memory_info()['draft_prefill_builder_workspace_bytes'],expected)
            self.assertEqual(obj.base_arguments[1]['draft_builder_workspace_bytes'],256<<20)
            obj.close();self.assertTrue(obj.base_closed)
    def test_invalid_prefill_limit_rejected_before_base_allocation(self):
        cls=load();before=FakeBase.constructions
        for value in (True,None,0,64<<20,129<<20,512<<20,float(256<<20)):
            with self.subTest(value=value),self.assertRaises(ValueError):cls(decoder(),'fixture',[],draft_prefill_builder_workspace_bytes=value)
        self.assertEqual(FakeBase.constructions,before)
    def test_t16_failure_with_new_budget_retires_base(self):
        def fail(*args,**kwargs):
            self.assertEqual(kwargs['builder_workspace_bytes'],256<<20)
            raise RuntimeError('T16 failure')
        with self.assertRaisesRegex(RuntimeError,'T16 failure'):load(fail)(decoder(),'fixture',[],draft_prefill_builder_workspace_bytes=256<<20)
        self.assertTrue(FakeBase.last.base_closed)

if __name__=='__main__':unittest.main()
