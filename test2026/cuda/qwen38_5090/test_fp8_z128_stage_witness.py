"""Synthetic CPU stage/counter/policy faults only, never hardware observations."""
from pathlib import Path
import copy,hashlib,json,unittest
from .test_gdn_chunk_stage_witness import fixture
from .fp8_z128_stage_layout import POLICY_VALUES,check_policy_report
from .fp8_z128_stage_replay import replay


def policy_report(value):
    invalid=value not in (None,'','0','1')
    return {'status':'fp8_z128_policy_control_observed','policy_value':value,'rejected':invalid,'error':'synthetic: GARNET_FP8_GDN_Z128_CANONICAL must be0 or1' if invalid else None,'graph':None if invalid else {'checkpoint_weight_precision':'modelopt_mixed','native_nvfp4_projection_weights':2,'native_fp8_projection_weights':3,'external_compressed_weight_inputs':5,'native_int4_g128_projection_weights':0,'native_fp8_m128_canonical_projections':1,'native_fp8_z128_canonical_projections':int(value=='1')},'graph_inference_executed':False,'model_or_performance_qualified':False}


class Controls(unittest.TestCase):
    def test_complete_synthetic_stage_replay_and_Z_counter_fault(self):
        report,files=fixture()
        for row in report['cases']:row['graph']['native_fp8_z128_canonical_projections']=int(row['tokens']==128)
        result=replay(copy.deepcopy(report),files.__getitem__);self.assertTrue(result['controlled_first128_all11_stages_exact'])
        for value in (False,1,2):
            bad=copy.deepcopy(report);bad['cases'][0]['graph']['native_fp8_z128_canonical_projections']=value
            with self.assertRaises(AssertionError):replay(bad,files.__getitem__)

    def test_fresh_process_policy_report_envelope_and_faults(self):
        for value in POLICY_VALUES:check_policy_report(policy_report(value),value)
        good=policy_report('1')
        for key,value in (('policy_value','0'),('graph_inference_executed',True),('model_or_performance_qualified',True),('rejected',True)):
            with self.assertRaises(AssertionError):check_policy_report(dict(good,**{key:value}),'1')
        for value in (False,0,2):
            bad=copy.deepcopy(good);bad['graph']['native_fp8_z128_canonical_projections']=value
            with self.assertRaises(AssertionError):check_policy_report(bad,'1')
        bad=policy_report('true');bad['error']='synthetic unrelated allocation failure'
        with self.assertRaises(AssertionError):check_policy_report(bad,'true')

    def test_probe_and_launchers_are_exact_preserved_derivatives(self):
        m=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()
        for name in ('fp8-z128-stage-probe-derivation.json','fp8-z128-stage-owner-derivation.json'):
            p=json.loads((m/'evidence'/name).read_bytes())
            for row in p['files']:
                raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
                for step in row['replacements']:
                    self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
                self.assertEqual(text.encode(),(m/row['target']).read_bytes());self.assertEqual(sha(text.encode()),row['target_sha256'])

if __name__=='__main__':unittest.main()
