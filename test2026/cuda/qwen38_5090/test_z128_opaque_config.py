import copy,unittest
from .z128_opaque_config import ATTRS,replay
class Controls(unittest.TestCase):
    def test_exact_inputs_and_incomplete_query_are_not_equivalence(self):
        left='00'*64;right='00'*34+'d3bb'+'00'*28
        attrs=[{'name':name,'requested_bytes':size,'attribute':i,'status':0,'written':size,'value_hex':(67 if i==0 else i).to_bytes(size,'little').hex()} for i,(name,size) in enumerate(ATTRS)]
        report={'status':'retained_opaque_documented_config_queries_collected','cublaslt_version':130400,'algorithm_substituted':False,'GPU_model_or_matmul_executed':False,'functional_equivalence_established':False,'descriptors':[{'label':label,'opaque':raw,'attributes':copy.deepcopy(attrs)} for label,raw in [('qualified_M32',left),('actual_failed_plugin_first_candidate',right)]]}
        q=replay(report,left,right);self.assertTrue(q['all_documented_attributes_equal']);self.assertEqual(q['differing_byte_offsets0'],[34,35]);self.assertFalse(q['functional_or_numerical_equivalence_established'])
        report['descriptors'][1]['attributes'][0]['status']=7;q=replay(report,left,right);self.assertFalse(q['all_documented_attributes_equal']);self.assertFalse(q['all_18_queries_successful'])
        report['descriptors'][1]['opaque']=left
        with self.assertRaises(AssertionError):replay(report,left,right)
if __name__=='__main__':unittest.main()
