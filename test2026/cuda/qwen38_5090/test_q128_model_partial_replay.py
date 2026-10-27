import copy,hashlib,struct,unittest
from .full_chunk_stage_layout import STAGES
from .q128_model_partial_replay import single_frame
from . import test_kv128_prefix_cache as graph_fixture
from .test_bf16_packed_state_gate import witness
sha=lambda b:hashlib.sha256(b).hexdigest()
def example():
    graphs=graph_fixture.Prefix().graphs(128)
    for k,row in graphs.items():row.update(native_full_chunk_stage_witnesses=int(k=='128'),native_fp8_q128_canonical_projections=16 if k=='128' else 0)
    files={};arrays={}
    for name,width in STAGES:
        raw=bytes(128*width*4);parts=[]
        for start in range(0,128*width,1048576):
            part=raw[start*4:(start+1048576)*4];n=name+str(start)+'.f32';files[n]=part;parts.append(dict(file=n,offset=start,words=len(part)//4,sha256=sha(part)))
        arrays[name]=dict(words=128*width,parts=parts,sha256=sha(raw))
    state=witness(1);state['slots']=[0];before=dict(state=state,controls={k:dict(words=w,sha256='a'*64) for k,w in (('0',655360),('2',128),('3',384),('4',128))});obs=dict(start=0,run=0,before=before,after=copy.deepcopy(before))
    r=dict(status='failed',error='AssertionError()',tokens=128,Q128_enabled=True,reference_profile='baseline',stage_capture_enabled=True,performance_admitted=False,original_reference_primary_equivalence_passed=False,graphs=graphs,stage_runs=31,stage_frames=[dict(start=0,count=128,arrays=arrays)],readonly_observations=[obs]);return r,files,copy.deepcopy(r)
class PartialReplayTests(unittest.TestCase):
    def test_single_complete_capture_never_accepts_repeat_decode_or_admission_metadata(self):
        r,files,off=example();self.assertEqual(single_frame(r,files.__getitem__,off)['count'],128)
        for field in ('ordinary_ids','ordinary_final','repeat_prefill','repeat_complete_prefill_state_vectors_exact','model_admission_passed'):
            bad=copy.deepcopy(r);bad[field]=False
            with self.assertRaises(AssertionError):single_frame(bad,files.__getitem__,off)
        bad=copy.deepcopy(r);bad['stage_frames'][0]['arrays']['qg']['repeat_sha256']=bad['stage_frames'][0]['arrays']['qg']['sha256']
        with self.assertRaises(AssertionError):single_frame(bad,files.__getitem__,off)
    def test_truncated_missing_parts_corruption_nonfinite_are_rejected(self):
        r,files,off=example();row=r['stage_frames'][0]['arrays']['qg']
        for field,value in (('words',True),('parts',row['parts'][:1]),('sha256','f'*64)):
            bad=copy.deepcopy(r);bad['stage_frames'][0]['arrays']['qg'][field]=value
            with self.assertRaises(AssertionError):single_frame(bad,files.__getitem__,off)
        filename=row['parts'][0]['file'];files[filename]=b'x'+files[filename][1:]
        with self.assertRaises(AssertionError):single_frame(r,files.__getitem__,off)
        raw=bytearray(files[filename]);struct.pack_into('<f',raw,0,float('nan'));files[filename]=bytes(raw);row['parts'][0]['sha256']=sha(raw);row['sha256']=sha(b''.join(files[p['file']] for p in row['parts']))
        with self.assertRaises(AssertionError):single_frame(r,files.__getitem__,off)
    def test_readonly_state_controls_counts_and_failure_identity_are_required(self):
        r,files,off=example()
        for field,value in (('status','passed'),('reference_profile','native'),('Q128_enabled',False),('stage_runs',62),('original_reference_primary_equivalence_passed',True)):
            bad=copy.deepcopy(r);bad[field]=value
            with self.assertRaises(AssertionError):single_frame(bad,files.__getitem__,off)
        bad=copy.deepcopy(r);bad['readonly_observations'][0]['after']['state']['components'][0]['row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):single_frame(bad,files.__getitem__,off)
        bad=copy.deepcopy(r);bad['readonly_observations'][0]['before']['controls']['0']['words']=1
        with self.assertRaises(AssertionError):single_frame(bad,files.__getitem__,off)
