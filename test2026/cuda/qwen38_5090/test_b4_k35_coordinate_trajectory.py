from pathlib import Path
import copy,struct,unittest
from .test_b4_k35_cache52 import layout
from .b4_k35_coordinate_trajectory import CoordinateSampler,checked_trajectory,compare,PAIR_WORD,codes
from .b4_k35_cache52 import CACHE_BYTES

def fixture(batch,position,first=45,code=0x3fa8):
    class IO:
        def zeros(self,shape):self.shape=shape;return object()
        def slice_into(self,state,start,part):assert start==55894016+PAIR_WORD
        def row(self,part,row):assert row==position;return b'\0\0'+struct.pack('<H',0 if self.step<first else code)
        def finite_f32_bytes(self,value):return value
    io=IO();s=CoordinateSampler(io,layout(),batch,position)
    for step in range(53):io.step=step;s.read(object(),step,[3968+step]*batch)
    raw=bytearray(CACHE_BYTES);raw[PAIR_WORD*4:PAIR_WORD*4+4]=bytes.fromhex(s.points[-1]['carrier_hex'])
    return s.result(),bytes(raw),io

class Trajectory(unittest.TestCase):
    def test_sample_one_existing_carrier_twice_correct_row_offset_and_full_cache_link(self):
        for batch,position in ((1,0),(4,2)):
            t,raw,io=fixture(batch,position);self.assertEqual(io.shape,[batch,1]);points=checked_trajectory(t,raw,batch,position);self.assertEqual(len(points),53);self.assertEqual(points[44]['selected_BF16_bits'],0);self.assertEqual(points[45]['selected_BF16_bits'],0x3fa8)
    def test_first_difference_derived_from_values_may_be_later_than_scheduled_writer(self):
        a,raw,_=fixture(1,0);b,other,_=fixture(4,2,code=0x3fa7);q=compare(checked_trajectory(a,raw,1,0),checked_trajectory(b,other,4,2));self.assertEqual(q['first_observed_selected_value_difference_step'],45)
        for p in b['points'][:48]:p['carrier_hex']=a['points'][p['step']]['carrier_hex'];p['repeat_carrier_hex']=[p['carrier_hex']]*2;p['BF16_codes']=codes(bytes.fromhex(p['carrier_hex']));p['selected_BF16_bits']=p['BF16_codes'][1]
        q=compare(checked_trajectory(a,raw,1,0),checked_trajectory(b,other,4,2));self.assertEqual(q['first_observed_selected_value_difference_step'],48);self.assertEqual(q['scheduled_decode_writer_step'],45)
    def test_carrier_and_selected_coordinate_differences_separated(self):
        a,raw,_=fixture(1,0);b,other,_=fixture(4,2);p=b['points'][5];p['carrier_hex']='00800000';p['repeat_carrier_hex']=[p['carrier_hex']]*2;p['BF16_codes']=[0x8000,0];p['selected_BF16_bits']=0
        q=compare(checked_trajectory(a,raw,1,0),checked_trajectory(b,other,4,2));self.assertEqual(q['first_observed_carrier_difference_step'],5);self.assertIsNone(q['first_observed_selected_value_difference_step'])
    def test_missing_step_wrong_row_slot_repeat_nonfinite_and_raw_link_rejected(self):
        for mutation in ('step','row','slot','repeat','NaN','cache'):
            t,raw,_=fixture(4,2)
            if mutation=='step':t['points'][7]['step']=6
            elif mutation=='row':t['physical_row']=1
            elif mutation=='slot':t['points'][45]['slots'][2]+=1
            elif mutation=='repeat':t['points'][7]['repeat_carrier_hex'][1]='00000001'
            elif mutation=='NaN':p=t['points'][7];p['carrier_hex']='0000c07f';p['repeat_carrier_hex']=[p['carrier_hex']]*2
            else:raw=raw[:PAIR_WORD*4]+b'\0\0\0\0'+raw[PAIR_WORD*4+4:]
            with self.assertRaises(AssertionError):checked_trajectory(t,raw,4,2)
    def test_live_sampler_rejects_out_of_order_slots_or_unstable_read(self):
        class IO:
            def zeros(self,shape):return None
            def slice_into(self,*args):pass
            def row(self,*args):return None
            def finite_f32_bytes(self,value):self.n=getattr(self,'n',0)+1;return bytes([self.n,0,0,0])
        for which in ('step','slots','unstable'):
            s=CoordinateSampler(IO(),layout(),4,2)
            with self.assertRaises(AssertionError):s.read(None,1 if which=='step' else 0,[3969 if which=='slots' else 3968]*4)
if __name__=='__main__':unittest.main()
