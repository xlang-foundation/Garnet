from pathlib import Path
import copy,hashlib,json,unittest
from .z128_factory_layer2 import joined_replay
class Controls(unittest.TestCase):
    def test_actual_native_enclosing_call_failure_and_silence(self):
        begin={'id':1,'phase':'begin','route':'fp8_z128','instance':'1','rows':6144,'columns':5120,'policy':0x5a313238,'context_present':False,'factory_called':False,'result':-1,'exception':False};end=dict(begin,phase='end',factory_called=True,result=1)
        factory={'id':1,'seq':1,'event':'begin','name':'factory','a':0,'b':0,'c':0,'opaque':''};rows=[factory,dict(factory,seq=2,event='api',name='cudaGetDevice',a=7),dict(factory,seq=3,event='reject',name='device_or_version'),dict(factory,seq=4,event='end',name='empty')]
        line=lambda prefix,row:prefix+json.dumps(row)+'\n'
        raw=line('GARNET_MODELOPT_INIT ',begin)+''.join(line('GARNET_Z128_FACTORY ',r) for r in rows)+line('GARNET_MODELOPT_INIT ',end)
        q=joined_replay(raw,True);self.assertTrue(q['actual_failure_guard_observed']);self.assertEqual(q['failures'][0]['guard']['name'],'device_or_version');self.assertFalse(q['Python_profile_stdout_chronology_used'])
        with self.assertRaises(AssertionError):joined_replay(raw,False)
        with self.assertRaises(AssertionError):joined_replay(raw.replace(line('GARNET_MODELOPT_INIT ',begin),''),True)
        self.assertFalse(joined_replay('',False)['actual_failure_guard_observed'])
    def test_exact_GPU_collector_derivatives(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();p=json.loads((m/'evidence/z128-factory-layer2-derivation.json').read_bytes())
        for row in p['files']:
            raw=(m/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for step in row['changes']:
                before=step['before'].encode();self.assertEqual(raw.count(before),step['count']);raw=raw.replace(before,step['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        self.assertFalse(p['numerical_collection_changed'])
if __name__=='__main__':unittest.main()
