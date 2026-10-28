"""Faults that remain internally consistent must still fail the retained oracle."""
import copy,hashlib,struct,unittest
from .test_mtp_prefill16_full4k import fixture
from .tiled_scores_model import replay_tiled_model

class TiledModelControls(unittest.TestCase):
    def test_default_and_tiled_exact_retained_oracle(self):
        for mode in (False,True):
            reference,files,cases=fixture();actual=copy.deepcopy(reference);actual['tiled_scores_enabled']=mode
            result=replay_tiled_model(actual,files.__getitem__,reference,files.__getitem__,cases)
            self.assertEqual(result['captured_states'],32);self.assertFalse(result['performance_admitted'])
    def test_shared_ordinary_mtp_state_drift_rejected(self):
        reference,files,cases=fixture();actual=copy.deepcopy(reference);actual['tiled_scores_enabled']=True
        for record in actual['cases']:
            if record['case_index']==0:
                for name in ('ordinary_final','candidate_final'):record[name]['logical']['components'][0]['row_sha256']=['a'*64]
        with self.assertRaises(AssertionError):replay_tiled_model(actual,files.__getitem__,reference,files.__getitem__,cases)
    def test_shared_finite_vector_drift_rejected(self):
        reference,files,cases=fixture();actual=copy.deepcopy(reference);actual['tiled_scores_enabled']=True;changed=dict(files)
        for record in actual['cases']:
            if record['case_index']==0:
                for name in ('ordinary_final','candidate_final'):
                    v=record[name]['vectors']['hidden'];raw=struct.pack('<f',99.)*5120;changed[v['file']]=raw;v['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(AssertionError):replay_tiled_model(actual,changed.__getitem__,reference,files.__getitem__,cases)
    def test_missing_route_or_nonboolean_rejected(self):
        for value in (None,0,1,'1'):
            reference,files,cases=fixture();actual=copy.deepcopy(reference);actual['tiled_scores_enabled']=value
            with self.assertRaises(AssertionError):replay_tiled_model(actual,files.__getitem__,reference,files.__getitem__,cases)

if __name__=='__main__':unittest.main()
