import json,unittest
from .b4_stage_profile_gate import admit,CORE
from test2026.cuda.qwen38_5090.native_stage_timing import StageRecorder
class GroupStageControls(unittest.TestCase):
    def events(self):
        out=[];clock=iter(range(1000));r=StageRecorder(out.append,lambda:next(clock))
        for ids in (['1'*32,'2'*32],['3'*32]):
            out.append(dict(event='study_profile_group',id=ids[0],request_ids=ids,requested_profile_flags=[True]*len(ids),real_rows=len(ids),physical_batch=4,measurement='host_wall_intervals_including_logging',CUDA_kernel_elapsed_claimed=False));r.start_request(ids[0])
            for name in sorted(CORE-{'worker_request'}):r.wrap(lambda:None,name)()
            r.finish_request()
        return out
    def raw(self,events):return b'\n'.join(json.dumps(dict(line=json.dumps(e))).encode() for e in events)
    def test_real_groups_have_one_native_clock_and_nonadditive_scope(self):
        q=admit(self.raw(self.events()),{'1'*32,'2'*32,'3'*32});self.assertEqual(q['real_request_IDs'],3);self.assertEqual(len(q['groups']),2);self.assertFalse(q['CUDA_kernel_elapsed_or_clean_serving_speed_qualified'])
    def test_duplicate_missing_unknown_or_nonfirst_group_IDs_rejected(self):
        for change in ('duplicate','missing','unknown','first'):
            e=self.events();expected={'1'*32,'2'*32,'3'*32}
            if change=='duplicate':e[0]['request_ids']=['1'*32,'1'*32]
            elif change=='missing':expected.add('4'*32)
            elif change=='unknown':expected.remove('2'*32)
            else:e[0]['id']='2'*32
            with self.assertRaises(AssertionError):admit(self.raw(e),expected)
    def test_wrong_clock_or_false_GPU_and_clean_speed_claims_rejected(self):
        for change in ('clock','kernel','speed'):
            e=self.events()
            if change=='clock':e[1]['clock_domain']='http_bridge'
            elif change=='kernel':e[0]['CUDA_kernel_elapsed_claimed']=True
            else:e[0]['clean_throughput_result_claimed']=True
            with self.assertRaises(AssertionError):admit(self.raw(e),{'1'*32,'2'*32,'3'*32})
    def test_sequence_time_and_nonfinite_timestamp_corruption_rejected(self):
        for change in ('sequence','time','nan'):
            e=self.events()
            if change=='sequence':e[2]['sequence']+=1
            elif change=='time':e[2]['native_monotonic_seconds']=-1
            else:e[2]['native_monotonic_seconds']=float('nan')
            with self.assertRaises(AssertionError):admit(self.raw(e),{'1'*32,'2'*32,'3'*32})
    def test_unbalanced_mismatched_parent_and_foreign_span_rejected(self):
        for change in ('unbalanced','parent','span','boolean_parent'):
            e=self.events()
            if change=='unbalanced':e.pop(3)
            elif change=='parent':e[2]['parent_span']=999
            elif change=='span':e[3]['span']=999
            else:e[2]['parent_span']=True
            with self.assertRaises(AssertionError):admit(self.raw(e),{'1'*32,'2'*32,'3'*32})
    def test_failed_incomplete_or_unprofiled_physical_group_rejected(self):
        for change in ('failed','coverage','flag'):
            e=self.events()
            if change=='failed':e[3]['call_succeeded']=False
            elif change=='coverage':
                for row in e:
                    if row.get('stage')=='target_program_t32':row['stage']='wrong_program'
            else:e[0]['requested_profile_flags'][1]=False
            with self.assertRaises(AssertionError):admit(self.raw(e),{'1'*32,'2'*32,'3'*32})
