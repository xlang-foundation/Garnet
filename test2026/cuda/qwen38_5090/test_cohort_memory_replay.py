"""CPU synthetic memory-window mutations; never measured GPU memory."""
import copy,unittest
from .cohort_memory_replay import memory_stats,replay_cohort_memory

def fixture():
    rows=[{'phase':'text-c1','monotonic':float(i),'used_bytes':100+i} for i in range(6)]
    summary=memory_stats(rows[:4]);summary['scope']='synthetic fixture only'
    requests=[{'status':'completed','submitted':.5,'completed':2.5}]
    return rows,summary,requests

class MemoryWindowControls(unittest.TestCase):
    def test_summary_snapshot_and_later_samples_are_both_retained(self):
        rows,summary,requests=fixture();before=copy.deepcopy((rows,summary,requests));result=replay_cohort_memory(rows,'text-c1',summary,requests)
        self.assertEqual(result['samples'],4);self.assertEqual(result['sampled_peak_adapter_used_bytes'],103)
        self.assertEqual(result['trailing_raw_phase_samples'],2);self.assertEqual(result['all_raw_phase_stats']['sampled_peak_adapter_used_bytes'],105)
        self.assertEqual(result['inside_request_wall_interval_sampled_peak_adapter_bytes'],102)
        self.assertEqual((rows,summary,requests),before)
    def test_recomputed_smaller_prefix_cannot_hide_the_end_of_measurement(self):
        rows,summary,requests=fixture();short=memory_stats(rows[:3])
        with self.assertRaisesRegex(ValueError,'bracket'):replay_cohort_memory(rows,'text-c1',short,requests)
    def test_peak_count_gap_and_interval_tampering_are_rejected(self):
        rows,summary,requests=fixture()
        for key,value in (('samples',True),('samples',7),('sampled_peak_adapter_used_bytes',102),('observed_sample_gap_max_seconds',2),('requested_interval_seconds',.1)):
            with self.subTest(key=key),self.assertRaises(ValueError):replay_cohort_memory(rows,'text-c1',dict(summary,**{key:value}),requests)
    def test_request_interval_extension_and_missing_phase_are_rejected(self):
        rows,summary,requests=fixture()
        with self.assertRaises(ValueError):replay_cohort_memory(rows,'text-c1',summary,[dict(requests[0],completed=4)])
        with self.assertRaises(ValueError):replay_cohort_memory(rows,'image-c1',summary,requests)
    def test_unordered_nonfinite_and_invalid_memory_rows_are_rejected(self):
        rows,summary,requests=fixture()
        for key,value in (('monotonic',float('nan')),('monotonic',1.5),('used_bytes',-1),('used_bytes',True)):
            changed=copy.deepcopy(rows);changed[-1][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):replay_cohort_memory(changed,'text-c1',summary,requests)

if __name__=='__main__':unittest.main()
