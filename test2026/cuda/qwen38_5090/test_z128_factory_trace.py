"""Exact real helper derivative and retained controls; no native or GPU execution."""
from pathlib import Path
import copy,hashlib,json,re,unittest
from unittest.mock import patch
from .z128_factory_trace import APIS,GUARDS,MODES,POLICIES,PREFIX,replay
M=Path(__file__).resolve().parent;ROOT=M.parents[2];E=M/'evidence';C=M/'native_z128_factory_trace_candidate';HELPER='src/backends/tensorrt/modelopt_fp8_z128_canonical.cpp';sha=lambda raw:hashlib.sha256(raw).hexdigest()

class Controls(unittest.TestCase):
    def test_exact_helper_composition_all_other_code_unchanged(self):
        p=json.loads((E/'z128-factory-trace-source-derivation.json').read_bytes());original=(C/'original-modelopt_fp8_z128_canonical.cpp.txt').read_bytes();self.assertEqual(sha(original),p['source_sha256']);actual=(ROOT/HELPER).read_bytes();self.assertIn(sha(actual),(p['source_sha256'],p['target_sha256']));raw=original
        for row in p['changes']:
            before=row['before'].encode();self.assertEqual(raw.count(before),row['count']);raw=raw.replace(before,row['after'].encode())
        self.assertEqual(raw,(ROOT/p['target']).read_bytes());self.assertEqual(sha(raw),p['target_sha256']);self.assertEqual(p['original_SDK_call_expressions'],17);self.assertEqual(p['new_SDK_calls'],0);self.assertFalse(p['algorithm_descriptors_workspace_wire_enqueue_changed'])
        # Removing wrappers restores all original SDK expressions; unchanged suffix includes RunMatmul.
        a=original.index(b'std::shared_ptr<ModelOptFP8Z128Context> Create');b=original.index(b'\nbool GetModelOptFP8Z128Info',a);x=raw.index(b'std::shared_ptr<ModelOptFP8Z128Context> Create');y=raw.index(b'\nbool GetModelOptFP8Z128Info',x)
        self.assertEqual(original[b:],raw[y:]);self.assertEqual(original[:a].replace(b'#include "modelopt_fp8_z128_canonical.h"',b'#include "modelopt_fp8_z128_canonical.h"\n#include "modelopt_fp8_z128_init_trace.h"'),raw[:x]);self.assertEqual((C/'modelopt_fp8_z128_canonical.h').read_bytes(),(ROOT/'src/backends/tensorrt/modelopt_fp8_z128_canonical.h').read_bytes())
        self.assertEqual(sha((C/'modelopt_fp8_z128_init_trace.h').read_bytes()),p['header_sha256']);self.assertEqual(len(APIS),len(GUARDS));self.assertEqual(len(MODES),48);self.assertEqual(len(POLICIES),10)
    def test_trace_fault_envelope_and_complete_opaque(self):
        begin={'id':1,'seq':1,'event':'begin','name':'factory','a':0,'b':0,'c':0,'opaque':''};rows=[begin,dict(begin,seq=2,event='api',name='cudaGetDevice',a=7),dict(begin,seq=3,event='reject',name='device_or_version'),dict(begin,seq=4,event='end',name='empty')]
        encode=lambda data:'\n'.join(PREFIX+json.dumps(r) for r in data)
        self.assertEqual(replay(encode(rows))['calls'][0]['reject']['name'],'device_or_version')
        for index,key,value in ((1,'seq',5),(1,'name','cublasLtCreate'),(2,'opaque','00'),(3,'name','created'),(0,'id',True)):
            bad=copy.deepcopy(rows);bad[index][key]=value
            with self.assertRaises(AssertionError):replay(encode(bad))
        with self.assertRaises(AssertionError):replay(encode(rows[:-1]))
    def test_retained_actual_plugin_failure_is_not_shadow_failure(self):
        p=json.loads((E/'modelopt-trace-layer2-attempt1-verified.json').read_bytes());self.assertTrue(p['actual_failed_plugin_route_observed']);self.assertFalse(p['helper_API_failure_cause_established']);self.assertEqual(len(p['actual_failed_initialize_returns']),1);row=p['actual_failed_initialize_returns'][0];self.assertEqual((row['route'],row['result'],row['factory_called'],row['context_present'],row['exception']),('fp8_z128',1,True,False,False));self.assertIsNone(row['begin_profile']);self.assertTrue(p['phases']['trace-on']['postfailure_helper']['created'])
    def test_original_policy_provenance_with_exact_factory_composition(self):
        from . import test_fp8_z128_native_source as original
        from .test_modelopt_init_trace import Controls as prior
        previous=original.source;snapshot=(C/'original-modelopt_fp8_z128_canonical.cpp.txt').read_bytes()
        with patch.object(original,'source',lambda name:snapshot if name==HELPER else previous(name)):prior('test_original_policy_provenance_through_exact_initialize_composition').debug()
    def test_original_integration_provenance_with_exact_factory_composition(self):
        from . import test_fp8_z128_native_integration as original
        from .test_modelopt_init_trace import Controls as prior
        previous=original.ROOT
        class OriginalSourceView:
            def __truediv__(self,name):return C/'original-modelopt_fp8_z128_canonical.cpp.txt' if name==HELPER else previous/name
        with patch.object(original,'ROOT',OriginalSourceView()):prior('test_original_prior_stage_integration_through_exact_initialize_composition').debug()
if __name__=='__main__':unittest.main()
