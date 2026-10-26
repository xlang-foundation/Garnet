"""CPU faults for bounded complete-cache reads, geometry, graph counts and state oracle."""
from pathlib import Path
import copy,hashlib,json,struct,unittest
from .fa3_prefix_cache_witness import cache_components,parts,read_cache,cache_difference,check_graphs,same_state,FIELDS,WORDS
from xModel.qwen35.vl_0_8b.kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
sha=lambda raw:hashlib.sha256(raw).hexdigest()
class Controls(unittest.TestCase):
    def test_layer3_geometry_and_bounded_reads(self):
        m=Path(__file__).resolve().parent
        for row in json.loads((m/'evidence/fa3-prefix-cache-witness-derivation.json').read_bytes())['derivations']:
            raw=(m/row['base']).read_bytes();self.assertEqual(sha(raw),row['base_sha256'])
            for change in row['changes']:
                before=change['before'].encode();self.assertEqual(raw.count(before),change['count']);raw=raw.replace(before,change['after'].encode())
            self.assertEqual(raw,(m/row['target']).read_bytes());self.assertEqual(sha(raw),row['target_sha256'])
        layout=build_bf16_low_batch_layout()['layers'];self.assertEqual(cache_components(layout),{'key':2482176,'value':4579328})
        self.assertEqual(parts(),[(0,1048576),(1048576,1048576)])
        for field in ('index','offset','state1_width','state2_width'):
            bad=copy.deepcopy(layout);bad[3][field]+=1
            with self.assertRaises(ValueError):cache_components(bad)
        data={'a.f32':bytes(4194304),'b.f32':bytes(4194304)};entry={'words':WORDS,'sha256':sha(bytes(8388608)),'parts':[{'file':name,'offset':i*1048576,'words':1048576,'sha256':sha(raw)} for i,(name,raw) in enumerate(data.items())]}
        self.assertEqual(len(read_cache(entry,data.__getitem__)),8388608)
        for field,value in (('file','../a.f32'),('offset',1),('words',True),('sha256','f'*64)):
            bad=copy.deepcopy(entry);bad['parts'][0][field]=value
            with self.assertRaises(AssertionError):read_cache(bad,data.__getitem__)
    def test_real_index_tail_and_nonfinite_rejection(self):
        baseline=bytes(WORDS*4);actual=bytearray(baseline);index=(2*4096+17)*256+39;actual[index*2:index*2+2]=struct.pack('<H',0x3f80)
        got=cache_difference(bytes(actual),baseline,128);self.assertEqual(got['changed_bf16_values'],1);self.assertEqual(got['changed_by_head'],[0,0,1,0]);self.assertEqual(got['first_difference'],{'head':2,'token':17,'dimension':39,'actual_bf16_bits':0x3f80,'reference_bf16_bits':0});self.assertEqual(got['maximum_absolute_difference'],1)
        for index,value in ((128*256,0x3f80),(0,0x7f80),(0,0x7fc0)):
            raw=bytearray(baseline);raw[index*2:index*2+2]=struct.pack('<H',value)
            with self.assertRaises(AssertionError):cache_difference(bytes(raw),baseline,128)
    def test_strict_graph_counts(self):
        from .prefill128_fp8_model import FIELD as OLD
        from .fp8_z128_full4k_model import FIELD as NEW
        self.assertEqual(FIELDS,(OLD,NEW))
        for tokens in (32,128):
            graphs={str(t):{field:48 if t==128 else 0 for field in FIELDS} for t in (4,tokens)};check_graphs(graphs,tokens)
            for field in FIELDS:
                for value in (True,1,None):
                    bad=copy.deepcopy(graphs);bad['4'][field]=value
                    with self.assertRaises(AssertionError):check_graphs(bad,tokens)
        with self.assertRaises(AssertionError):check_graphs({},True)
    def test_retained_state_and_raw_vector_faults(self):
        from .test_prefill128_model import data
        actual,reference,files,_=data();a=actual['ordinary_observations'][0]['ordinary_prefill'];b=reference['cases'][0]['ordinary_prefill']
        self.assertTrue(same_state(a,b,files.__getitem__,files.__getitem__))
        bad=copy.deepcopy(a);bad['logical']['components'][0]['row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):same_state(bad,b,files.__getitem__,files.__getitem__)
        badfiles=dict(files);name=a['vectors']['hidden']['file'];badfiles[name]=bytes(20480)
        if badfiles[name]==files[name]:badfiles[name]=struct.pack('<5120f',*([1.0]*5120))
        with self.assertRaises(AssertionError):same_state(a,b,badfiles.__getitem__,files.__getitem__)
if __name__=='__main__':unittest.main()
