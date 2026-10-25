"""CPU shared-drift faults must fail even if every permutation agrees internally."""
import copy,hashlib,struct,unittest
from .test_bf16_batch8_replay import fixture
from .paged_batch8_model import replay_preserved

def candidate():
    reference,files,_=fixture();actual=copy.deepcopy(reference)
    for c in actual['cases']:
        lengths=[len(r['prompt_ids']) for r in c['inputs']];n=max(lengths)
        c['prompt_cache']={'physical_batch':8,'tokens':n,'lengths':lengths,'page_size':128,'pack_calls':(n+127)//128,'cached_pages':[0],'peak_cached_packed_logical_bytes':min(n,128)*8*20480}
    return actual,reference,files

class PagedBatch8Controls(unittest.TestCase):
    def test_all_nine_synthetic_model_cases_and58vectors(self):
        actual,reference,files=candidate();r=replay_preserved(actual,files.__getitem__,reference,files.__getitem__)
        self.assertEqual(r['complete_states'],29);self.assertEqual(r['raw_vectors'],58);self.assertFalse(r['performance_admitted'])
    def test_shared_state_drift_that_preserves_permutation_is_rejected(self):
        actual,reference,files=candidate()
        for c in actual['cases']:c['final_state']['logical']['components'][0]['row_sha256']=['f'*64]*8
        with self.assertRaises(AssertionError):replay_preserved(actual,files.__getitem__,reference,files.__getitem__)
    def test_shared_finite_vector_drift_with_updated_hash_is_rejected(self):
        actual,reference,files=candidate();changed=dict(files)
        for c in actual['cases']:
            entry=c['final_state']['vectors']['hidden'];raw=struct.pack('<f',99.)*40960;changed[entry['file']]=raw;entry['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaises(AssertionError):replay_preserved(actual,changed.__getitem__,reference,files.__getitem__)
    def test_case_missing_and_unbounded_materialization_rejected(self):
        for kind in ('missing','cache'):
            actual,reference,files=candidate()
            if kind=='missing':actual['cases'].pop()
            else:actual['cases'][-1]['prompt_cache']['peak_cached_packed_logical_bytes']=41943041
            with self.subTest(kind=kind),self.assertRaises(AssertionError):replay_preserved(actual,files.__getitem__,reference,files.__getitem__)

if __name__=='__main__':unittest.main()
