"""Fault tests for observation replay and exact-source isolation, no measured GPU data."""
from pathlib import Path
import copy,hashlib,json,unittest
from .fp8_z128_context_audit import replay
M=Path(__file__).resolve().parent;REPO=M.parents[2];OPAQUE='43'+'00'*63

def report(created=True):
    candidate={'state':0,'workspace':0,'opaque':OPAQUE,'post_context_ID_status':0,'post_context_ID':67,'post_context_ID_bytes':4}
    return {'status':'actual_Z128_context_observed','created':created,'info_okay':created,'device':0,'device_status':0,'cuda_error_before':0,'cuda_error_after':0,'matmul_calls':0,'algorithm_substituted':False,'gpu_model_executed':False,'model_or_performance_qualified':False,'heuristics':[{'api_status':0,'requested':8,'found':1,'candidates':[candidate]}],'helper_info':{'algorithm':67,'device':0,'opaque':OPAQUE,'required_workspace':0},'config_calls':[{'api_status':0,'attribute':0,'requested':4,'written':4,'value_hex':'43000000'}] if created else [],'check_calls':[{'api_status':0,'state':0,'workspace':0}] if created else []}

class ContextAudit(unittest.TestCase):
    def test_created_and_faulted_replay(self):
        good=report();self.assertTrue(replay(good,OPAQUE)['created'])
        for key,value in [('device',1),('cuda_error_after',1),('matmul_calls',1),('algorithm_substituted',True),('gpu_model_executed',True),('info_okay',False)]:
            bad=copy.deepcopy(good);bad[key]=value
            with self.assertRaises(AssertionError):replay(bad,OPAQUE)
        for key,value in [('api_status',1),('state',1),('workspace',4194305)]:
            bad=copy.deepcopy(good);bad['check_calls'][0][key]=value
            with self.assertRaises(AssertionError):replay(bad,OPAQUE)

    def test_first_success_rejection_and_later_exact_candidate(self):
        bad=report(False);c=bad['heuristics'][0]['candidates'][0];c['opaque']='44'+'00'*63
        bad['heuristics'][0]['candidates'].append(copy.deepcopy(report()['heuristics'][0]['candidates'][0]));bad['heuristics'][0]['found']=2
        result=replay(bad,OPAQUE);self.assertEqual(result['expected_opaque_candidate_indices'],[1]);self.assertIn('rejected_first_successful_opaque',result['boundary'])
        self.assertFalse(result['full_Garnet_plugin_failure_directly_instrumented'])
        malformed=copy.deepcopy(bad);malformed['config_calls']=[{}]
        with self.assertRaises(AssertionError):replay(malformed,OPAQUE)

    def test_complete_candidates_required(self):
        for field,value in [('found',2),('found',True),('requested',9),('api_status',1)]:
            bad=report();bad['heuristics'][0][field]=value
            with self.assertRaises(AssertionError):replay(bad,OPAQUE)
        for field,value in [('opaque','43'),('post_context_ID_bytes',8),('post_context_ID_status',1)]:
            bad=report();bad['heuristics'][0]['candidates'][0][field]=value
            with self.assertRaises(AssertionError):replay(bad,OPAQUE)

    def test_actual_helper_observation_without_substitution(self):
        raw=(M/'fp8_z128_context_audit.cpp').read_text();self.assertEqual(raw.count('#include "modelopt_fp8_z128_canonical.cpp"'),1)
        for api,wrapper in [('cublasLtMatmulAlgoGetHeuristic','AuditHeuristic'),('cublasLtMatmulAlgoConfigGetAttribute','AuditConfig'),('cublasLtMatmulAlgoCheck','AuditCheck'),('cublasLtMatmul','AuditMatmul')]:
            self.assertIn('#define '+api+' '+wrapper,raw);self.assertIn('::'+api+'(',raw);self.assertIn('#undef '+api,raw)
        self.assertNotIn('cublasLtMatmulAlgoInit',raw);self.assertNotIn('cublasLtMatmulAlgoConfigSetAttribute',raw)
        self.assertNotIn('RunModelOptFP8Z128Matmul(',raw);self.assertIn('CREATE_NEW',raw)
        # Existing closed origin pins the exact actual helper whose first-success guard is observed.
        origin=json.loads((M/'evidence/fp8-z128-native-source-derivation.json').read_bytes())
        self.assertIn('memcmp',(REPO/'src/backends/tensorrt/modelopt_fp8_z128_canonical.cpp').read_text() if (REPO/'src/backends/tensorrt/modelopt_fp8_z128_canonical.cpp').exists() else (M/'native_fp8_z128_candidate/modelopt_fp8_z128_canonical.cpp').read_text())

    def test_exact_probe_owner_and_host_derivations(self):
        for name in ('fp8-z128-context-audit-derivation.json','fp8-z128-ctypes-bound-derivation.json','fp8-z128-ctypes-bound2-derivation.json','fp8-z128-context-audit-analyzer-derivation.json'):
            d=json.loads((M/'evidence'/name).read_bytes())
            for row in d['derivations']:
                raw=(M/row['base']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['base_sha256'])
                for change in row['changes']:
                    a=change['before'].encode();self.assertEqual(raw.count(a),1);raw=raw.replace(a,change['after'].encode())
                self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])

if __name__=='__main__':unittest.main()
