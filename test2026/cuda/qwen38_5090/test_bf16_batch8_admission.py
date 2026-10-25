"""CPU complete-row/witness permutation tests; synthetic hashes, no GPU claims."""
import copy,hashlib,unittest
from .bf16_batch8_admission import ORDERS,one_row,compare_permuted,compare_vector_rows
from .test_bf16_packed_state_gate import witness
from .bf16_packed_state_gate import same_packed_bf16_row

def unique():
    value=witness(8);value['slots']=list(range(8))
    digest=lambda text:hashlib.sha256(text.encode()).hexdigest()
    for key in ('physical_row_sha256','logical_row_sha256'):value[key]=[digest(key+str(i)) for i in range(8)]
    for n,c in enumerate(value['components']):c['row_sha256']=[digest(str(n)+':'+str(i)) for i in range(8)]
    return value

def permute(value,order):
    result=copy.deepcopy(value)
    for key in ('slots','physical_row_sha256','logical_row_sha256'):result[key]=[value[key][i] for i in order]
    for target,source in zip(result['components'],value['components']):target['row_sha256']=[source['row_sha256'][i] for i in order]
    return result

class AdmissionControls(unittest.TestCase):
    def test_rotated_full_state_is_equal_only_after_inverse_mapping(self):
        left=unique();right=permute(left,ORDERS[1]);before=copy.deepcopy((left,right))
        self.assertTrue(compare_permuted(left,right,ORDERS[1]));self.assertFalse(compare_permuted(left,right,ORDERS[0]));self.assertEqual((left,right),before)
    def test_each_row_and_each_component_remain_required(self):
        left=unique()
        for row in range(8):
            right=copy.deepcopy(left);right['components'][-1]['row_sha256'][row]='f'*64
            self.assertFalse(compare_permuted(left,right,ORDERS[0]))
    def test_invalid_unselected_metadata_cannot_hide_behind_an_earlier_difference(self):
        left=unique();right=copy.deepcopy(left);right['slots'][0]+=1;right['components'][-1]['row_sha256'][7]='invalid'
        with self.assertRaises(ValueError):compare_permuted(left,right,ORDERS[0])
        with self.assertRaises(ValueError):one_row(right,0)
    def test_old_b1_b4_gate_and_default_admission_remain_unchanged(self):
        self.assertTrue(same_packed_bf16_row(witness(1),0,witness(4),3))
        with self.assertRaises(ValueError):same_packed_bf16_row(witness(8),0,witness(8),0)
        self.assertTrue(same_packed_bf16_row(one_row(unique(),7),0,one_row(unique(),7),0))
    def test_malformed_geometry_and_non_permutation_rows_are_rejected(self):
        for order in (None,[0]*8,[True]+list(range(1,8)),list(range(7)),list(range(1,9))):
            with self.assertRaises(ValueError):compare_permuted(unique(),unique(),order)
        for change in ({'physical_words_per_row':173932544},{'reference_bf16_roundtrip_checked':True},{'slots':[0]*4}):
            right=unique();right.update(change)
            with self.assertRaises(ValueError):compare_permuted(unique(),right,ORDERS[0])
    def test_raw_vector_all_bytes_and_row_order_are_checked(self):
        width=5120*4;left=b''.join(bytes([i])*width for i in range(8));right=b''.join(left[i*width:(i+1)*width] for i in ORDERS[1])
        self.assertTrue(compare_vector_rows(left,right,ORDERS[1]));self.assertFalse(compare_vector_rows(left,right,ORDERS[0]))
        changed=bytearray(right);changed[-1]^=1;self.assertFalse(compare_vector_rows(left,changed,ORDERS[1]))
        with self.assertRaises(ValueError):compare_vector_rows(left,right[:-1],ORDERS[1])

if __name__=='__main__':unittest.main()
