from pathlib import Path
import copy,hashlib,json,unittest
from .modelopt_trace_layer2 import trace_profiles
class Controls(unittest.TestCase):
    def test_actual_event_profile_pairing_failure_and_silence(self):
        first={'id':1,'phase':'begin','route':'fp8','instance':'1','rows':5120,'columns':6144,'policy':1,'context_present':False,'factory_called':False,'result':-1,'exception':False};last=dict(first,phase='end',context_present=True,factory_called=True,result=0)
        second=dict(first,id=2,route='fp8_m128',policy=128,instance='2');failure=dict(second,phase='end',factory_called=True,result=1)
        def line(row):return 'GARNET_MODELOPT_INIT '+json.dumps(row)+'\n'
        raw='GARNET_TRACE_PROFILE_BEGIN 32\n'+line(first)+line(last)+'GARNET_TRACE_PROFILE_READY 32\nGARNET_TRACE_PROFILE_BEGIN 128\n'+line(second)+line(failure)
        q=trace_profiles(raw,True,False);self.assertTrue(q['actual_failed_plugin_route_observed']);self.assertEqual(q['failed_initialize_returns'][0]['begin_profile'],128)
        with self.assertRaises(AssertionError):trace_profiles(raw,True,True)
        with self.assertRaises(AssertionError):trace_profiles(raw,False,False)
        q=trace_profiles('GARNET_TRACE_PROFILE_BEGIN 32\nGARNET_TRACE_PROFILE_READY 32\nGARNET_TRACE_PROFILE_BEGIN 128\nGARNET_TRACE_PROFILE_READY 128\n',False,True);self.assertEqual(q['events'],0)
        with self.assertRaises(AssertionError):trace_profiles(raw+'GARNET_TRACE_PROFILE_BEGIN 128\n',True,False)
    def test_exact_probe_owner_launcher_derivations(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();p=json.loads((m/'evidence/modelopt-trace-layer2-derivation.json').read_bytes())
        for row in p['files']:
            raw=(m/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                before=change['before'].encode();self.assertEqual(raw.count(before),change['count']);raw=raw.replace(before,change['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        self.assertFalse(p['native_or_numerical_math_changed']);self.assertEqual(p['candidate_sources'],38)
