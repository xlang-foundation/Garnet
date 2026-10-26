import copy,hashlib,struct,unittest
from .b4_cache_mode_characterization import characterize_case,compare_capture

def fixture(batch):
    files={};vectors={}
    raw=struct.pack('<f',0)*batch*5120
    for kind in ('hidden','normalized'):
        name=kind+'.f32';files[name]=raw;vectors[kind]={'file':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    components=[]
    for layer in range(64):
        for role in (1,2):components.append({'layer':layer,'role':role,'kind':'full_attention' if layer%4==3 else 'linear_attention','logical_words':4194304 if layer%4==3 else (40960 if role==1 else 786432),'row_sha256':['a'*64]*batch})
    capture={'logical':{'physical_words_per_row':173932544,'logical_words_per_row':173932544,'slots':[104]*batch,'logical_row_sha256':['a'*64]*batch,'components':components},'vectors':vectors}
    case={'actual_limit':8,'group':'short-mixed','repeat':0,'order':list(range(8)),'inputs':[{'prompt_ids':[1,2]}]*batch,'generated_ids':[[1]*8 for _ in range(batch)],'text':['x']*batch,
          **{key:copy.deepcopy(capture) for key in ('reset_state','prefill_state','final_state')}}
    return case,files

class Controls(unittest.TestCase):
    def test_identical_actual_case_and_packed_logical_carrier(self):
        a,fa=fixture(4);b,fb=fixture(8)
        for phase in ('reset_state','prefill_state','final_state'):a[phase]['logical']['physical_words_per_row']=106823680
        self.assertTrue(characterize_case(a,b,[0,1,2,3],fa.__getitem__,fb.__getitem__)['all_observed_ids_states_vectors_exact'])
    def test_id_and_state_differences_are_retained_without_admission(self):
        a,fa=fixture(4);b,fb=fixture(8);a['generated_ids'][1][0]=2
        a['prefill_state']['logical']['components'][6]['row_sha256'][2]='b'*64
        r=characterize_case(a,b,[0,1,2,3],fa.__getitem__,fb.__getitem__)
        self.assertEqual(r['generated_ids'][1]['first_difference_zero_based'],0)
        self.assertEqual(r['phases']['prefill_state']['different_components'][0]['layer'],3)
        self.assertFalse(r['all_observed_ids_states_vectors_exact']);self.assertFalse(r['cause_or_harmlessness_proven'])
    def test_wrong_input_or_control_geometry_is_rejected(self):
        a,fa=fixture(4);b,fb=fixture(8)
        for key,value in (('actual_limit',128),('inputs',[{}]*4),('generated_ids',[[True]*8]*4)):
            c=copy.deepcopy(a);c[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):characterize_case(c,b,[0,1,2,3],fa.__getitem__,fb.__getitem__)
    def test_bad_vector_hash_bytes_and_nonfinite_values_are_rejected(self):
        a,fa=fixture(4);b,fb=fixture(8)
        for bad in (fa['hidden.f32'][:-4],struct.pack('<f',float('nan'))*4*5120):
            files=dict(fa,**{'hidden.f32':bad})
            with self.assertRaises(AssertionError):characterize_case(a,b,[0,1,2,3],files.__getitem__,fb.__getitem__)
    def test_incomplete_component_coverage_or_slot_change_is_rejected(self):
        a,fa=fixture(4);b,fb=fixture(8)
        for key,value in (('slots',[103]*4),('components',a['prefill_state']['logical']['components'][:-1])):
            c=copy.deepcopy(a);c['prefill_state']['logical'][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):characterize_case(c,b,[0,1,2,3],fa.__getitem__,fb.__getitem__)

if __name__=='__main__':unittest.main()
