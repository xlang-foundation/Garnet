"""Scientific window selection must exclude cold/foreign work and fail closed."""
import copy,unittest
from cuda_trace_windows import select_window

class WindowTests(unittest.TestCase):
    def setUp(self):
        self.copies=[dict(copyKind=1,bytes=35892,start=1,end=10,globalPid=4,deviceId=0),
            dict(copyKind=1,bytes=35908,start=30,end=40,globalPid=4,deviceId=0)]
        self.kernel=dict(start=11,end=29,globalPid=4,deviceId=0)
    def select(self,kernels,copies=None):return select_window(kernels,self.copies if copies is None else copies,35892,35908)
    def test_exact_boundaries_and_exclusions(self):
        exact=dict(self.kernel,start=10,end=30)
        cold=dict(self.kernel,start=0,end=9);late=dict(self.kernel,start=31,end=35)
        foreign_pid=dict(self.kernel,globalPid=5);foreign_device=dict(self.kernel,deviceId=1)
        self.assertEqual(self.select([cold,exact,late,foreign_pid,foreign_device]),(10,30,[exact]))
    def test_duplicate_or_missing_markers_rejected(self):
        for copies in (self.copies[:1],self.copies+[copy.copy(self.copies[0])]):
            with self.subTest(copies=copies),self.assertRaises(AssertionError):self.select([self.kernel],copies)
    def test_crossing_kernel_rejected(self):
        for a,b in ((9,11),(29,31),(9,31)):
            with self.subTest(span=(a,b)),self.assertRaises(AssertionError):self.select([dict(self.kernel,start=a,end=b)])
    def test_marker_device_or_process_mismatch_rejected(self):
        for key in ('deviceId','globalPid'):
            copies=copy.deepcopy(self.copies);copies[1][key]+=1
            with self.subTest(field=key),self.assertRaises(AssertionError):self.select([self.kernel],copies)
    def test_reversed_marker_order_rejected(self):
        copies=copy.deepcopy(self.copies);copies[1]['start']=8
        with self.assertRaises(AssertionError):self.select([self.kernel],copies)
    def test_wrong_copy_kind_not_a_marker(self):
        copies=copy.deepcopy(self.copies);copies[0]['copyKind']=2
        with self.assertRaises(AssertionError):self.select([self.kernel],copies)

if __name__=='__main__':unittest.main()
