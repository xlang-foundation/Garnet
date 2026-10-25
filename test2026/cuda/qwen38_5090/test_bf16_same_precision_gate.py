"""Guard against physical-layout equality and incomplete logical witness admission."""
import copy,hashlib,unittest
from test2026.cuda.qwen38_5090.bf16_same_precision_gate import same_logical_state


def states():
    hashes=lambda prefix:[hashlib.sha256((prefix+str(row)).encode()).hexdigest() for row in range(4)]
    ref={'physical_words_per_row':173932544,'logical_words_per_row':173932544,'reference_bf16_roundtrip_checked':True,
         'slots':[41,104,41,104],'physical_row_sha256':hashes('logical'),'logical_row_sha256':hashes('logical'),'components':[]}
    for layer in range(64):
        full=layer%4==3
        for role in (1,2):ref['components'].append({'layer':layer,'kind':'full_attention' if full else 'linear_attention','role':role,
            'logical_words':4194304 if full else (40960 if role==1 else 786432),'row_sha256':hashes(str((layer,role)))})
    packed=copy.deepcopy(ref);packed.update(physical_words_per_row=106823680,reference_bf16_roundtrip_checked=False,physical_row_sha256=hashes('packed'))
    return packed,ref


class SamePrecisionGateTests(unittest.TestCase):
    def test_matching_logical_content_admits_different_physical_bytes(self):
        packed,ref=states();before=copy.deepcopy((packed,ref))
        self.assertNotEqual(packed['physical_row_sha256'],ref['physical_row_sha256']);self.assertTrue(same_logical_state(packed,ref));self.assertEqual((packed,ref),before)

    def test_component_global_or_slot_difference_never_admits(self):
        for kind in ('component','global','slot'):
            packed,ref=states()
            if kind=='component':packed['components'][-1]['row_sha256'][3]='f'*64
            elif kind=='global':packed['logical_row_sha256'][3]='f'*64
            else:packed['slots'][3]+=1
            with self.subTest(kind=kind):self.assertFalse(same_logical_state(packed,ref))

    def test_missing_or_invalid_metadata_rejected(self):
        for kind in ('missing','duplicate','width','index_bool','role','hash','slots_bool','flag','physical'):
            packed,ref=states()
            if kind=='missing':packed['components'].pop()
            elif kind=='duplicate':packed['components'][4]=copy.deepcopy(packed['components'][3])
            elif kind=='width':packed['components'][7]['logical_words']//=2
            elif kind=='index_bool':packed['components'][2]['layer']=True
            elif kind=='role':packed['components'][0]['role']=2
            elif kind=='hash':packed['logical_row_sha256'][0]='bad'
            elif kind=='slots_bool':packed['slots'][0]=True
            elif kind=='flag':ref['reference_bf16_roundtrip_checked']=1
            else:packed['physical_words_per_row']=173932544
            with self.subTest(kind=kind),self.assertRaises(ValueError):same_logical_state(packed,ref)

if __name__=='__main__':unittest.main()
