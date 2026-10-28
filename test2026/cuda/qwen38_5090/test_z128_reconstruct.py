from pathlib import Path
import copy,hashlib,json,unittest
from .z128_reconstruct import APIS,FIELDS,MODES,PREFIX,replay,sdk_case
M=Path(__file__).resolve().parent;ROOT=M.parents[2];C=M/'native_z128_reconstruct_candidate';sha=lambda raw:hashlib.sha256(raw).hexdigest()
class Controls(unittest.TestCase):
    def test_separate_exact_derivative_and_public_configuration_provenance(self):
        p=json.loads((M/'evidence/z128-reconstruct-source-derivation.json').read_bytes());self.assertEqual(len(p['files']),3);self.assertTrue(p['legacy_factory_and_RunMatmul_bytes_unchanged']);self.assertFalse(p['old_plugin_or_default_route_changed'])
        for row in p['files']:
            raw=(ROOT/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for step in row['changes']:
                before=step['before'].encode();self.assertEqual(raw.count(before),step['count']);raw=raw.replace(before,step['after'].encode())
            self.assertEqual(raw,(ROOT/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        old=(M/'native_z128_factory_trace_candidate/modelopt_fp8_z128_canonical.cpp').read_bytes();new=(C/'modelopt_fp8_z128_canonical.cpp').read_bytes();marker=b'std::shared_ptr<ModelOptFP8Z128Context> CreateModelOptFP8Z128Context(){';body=(M/'z128_reconstruct_body.inc').read_bytes();self.assertEqual(sha(body),p['body_sha256']);self.assertEqual(old[old.index(marker):old.index(b'bool GetModelOptFP8Z128Info')],new[new.index(marker):new.index(body)]);self.assertEqual(old[old.index(b'bool GetModelOptFP8Z128Info'):],new[new.index(b'bool GetModelOptFP8Z128Info'):]);self.assertNotIn(b'memcmp',body);self.assertNotIn(b'GetHeuristic',body);self.assertNotIn(b'0x43',body)
        q=json.loads((M/'evidence/z128-opaque-config-audit-attempt2-verified.json').read_bytes())['replay'];self.assertTrue(q['all_18_queries_successful']);self.assertTrue(q['all_documented_attributes_equal']);self.assertFalse(q['functional_or_numerical_equivalence_established']);self.assertEqual([dict((name,value) for name,value,_ in FIELDS)]*2,q['values']);self.assertEqual(len(APIS),32);self.assertEqual(len(MODES),91)
    def test_CPU_driver_and_interfaces_exact_derivative(self):
        p=json.loads((M/'evidence/z128-reconstruct-cpu-derivation.json').read_bytes())
        for row in p['files']:
            raw=(ROOT/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256'])
            for step in row['changes']:
                if step.get('operation')=='append':raw+=step['after'].encode()
                else:before=step['before'].encode();self.assertEqual(raw.count(before),step['count']);raw=raw.replace(before,step['after'].encode())
            self.assertEqual(raw,(ROOT/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        self.assertEqual(sha((C/'cpu_sdk/cuda_runtime_api.h').read_bytes()),p['cuda_interface_sha256'])
    def test_complete_trace_and_configuration_faults_fail(self):
        seed={'id':1,'seq':1,'event':'begin','name':'factory','a':0,'b':0,'c':0,'opaque':''};rows=[seed]
        for name in APIS:rows.append(dict(seed,event='api',name=name,a=130400 if name=='cublasLtGetVersion' else 0))
        rows +=[dict(seed,event='configuration',name=name,a=value,b=size) for name,value,size in FIELDS]
        rows +=[dict(seed,event='algorithm',name='configured',opaque='00'*64),dict(seed,event='check',name='M32'),dict(seed,event='check',name='M128'),dict(seed,event='accept',name='created'),dict(seed,event='end',name='created')]
        for i,row in enumerate(rows):row['seq']=i+1
        encode=lambda values:'\n'.join(PREFIX+json.dumps(r) for r in values)
        self.assertEqual(replay(encode(rows))['calls'][0]['API_calls'],32)
        for i,key,value in ((1,'name','cublasLtCreate'),(1,'a',7),(33,'a',35),(34,'b',2),(-3,'b',4194305),(-1,'name','cached')):
            bad=copy.deepcopy(rows);bad[i][key]=value
            with self.assertRaises(AssertionError):replay(encode(bad))
        with self.assertRaises(AssertionError):replay(encode(rows[:-1]))
if __name__=='__main__':unittest.main()
