"""CPU source and actual-profile report faults; no native runtime/GPU."""
from pathlib import Path
import hashlib,json,unittest
from .gdn_chunk_stage_build_contract import profile_contract

class Controls(unittest.TestCase):
    def test_native_cpu_profile_report_counts_types_and_scope(self):
        good={'status':'gdn_chunk_stage_cpu_profile_passed','profiles':4,'profile_rejections':50,'flag_rejections':6,'legacy_conflict_rejections':1,'legacy_default_unchanged':True,'local_or_remote_gpu_calls':0,'native_frontend_exercised':False,'primary_witness_qualified':False,'model_or_performance_qualified':False};profile_contract(good)
        for key,value in good.items():
            bad=dict(good);del bad[key]
            with self.assertRaises(AssertionError):profile_contract(bad)
            bad=dict(good);bad[key]=not value if type(value) is bool else value+1 if type(value) is int else 'wrong'
            with self.assertRaises(AssertionError):profile_contract(bad)
        with self.assertRaises(AssertionError):profile_contract(dict(good,profiles=True))

    def test_build_method_derivations_are_exact(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest();r=json.loads((m/'evidence/gdn-chunk-stage-native-build-derivation.json').read_bytes())
        for row in r['files']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])

if __name__=='__main__':unittest.main()
