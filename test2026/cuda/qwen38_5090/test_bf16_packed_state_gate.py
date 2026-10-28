"""CPU representation/selected-row controls for the packed-versus-packed gate."""
import copy,unittest
from .bf16_packed_state_gate import same_packed_bf16_row
def witness(batch):
 return {'physical_words_per_row':106823680,'logical_words_per_row':173932544,'reference_bf16_roundtrip_checked':False,'slots':[20]*batch,'physical_row_sha256':['a'*64]*batch,'logical_row_sha256':['b'*64]*batch,'components':[{'layer':i//2,'role':i%2+1,'kind':'full_attention' if i//2%4==3 else 'linear_attention','logical_words':4194304 if i//2%4==3 else (40960 if i%2==0 else 786432),'row_sha256':['c'*64]*batch} for i in range(128)]}
class PackedRowGate(unittest.TestCase):
 def test_b1_b4_and_selected_rows(self):
  self.assertTrue(same_packed_bf16_row(witness(1),0,witness(4),3))
 def test_unselected_values_can_differ(self):
  a=witness(4);b=copy.deepcopy(a);b['slots'][1]=21;b['components'][7]['row_sha256'][1]='d'*64
  self.assertTrue(same_packed_bf16_row(a,0,b,0));self.assertFalse(same_packed_bf16_row(a,1,b,1))
 def test_selected_component_difference_is_not_hidden(self):
  a=witness(1);b=copy.deepcopy(a);b['components'][7]['row_sha256'][0]='d'*64;self.assertFalse(same_packed_bf16_row(a,0,b,0))
 def test_physical_and_logical_hashes_are_both_required(self):
  for key in ('physical_row_sha256','logical_row_sha256'):
   a=witness(1);b=copy.deepcopy(a);b[key][0]='d'*64;self.assertFalse(same_packed_bf16_row(a,0,b,0))
 def test_reference_representation_is_rejected(self):
  for change in ({'physical_words_per_row':173932544},{'reference_bf16_roundtrip_checked':True},{'logical_words_per_row':True}):
   b=witness(4);b.update(change)
   with self.assertRaises(ValueError):same_packed_bf16_row(witness(4),0,b,0)
 def test_invalid_unselected_metadata_cannot_be_hidden(self):
  for kind in ('hash','slot','order'):
   b=witness(4)
   if kind=='hash':b['components'][7]['row_sha256'][3]='invalid'
   elif kind=='slot':b['slots'][3]=True
   else:b['components'][127]['role']=1
   with self.assertRaises(ValueError):same_packed_bf16_row(witness(4),0,b,0)
 def test_row_types_and_boundaries(self):
  for row in (True,-1,1):
   with self.assertRaises(ValueError):same_packed_bf16_row(witness(1),row,witness(4),0)
if __name__=='__main__':unittest.main()
