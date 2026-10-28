import hashlib,json,unittest
from .fa3_projection_witness_source import sources,validate,M
class Controls(unittest.TestCase):
    def test_exact_additive_branch_keeps_all_qualified_ordinary_code(self):
        old,new,p=sources();self.assertTrue(validate(old,new,p));self.assertEqual(len(p['changes']),2)
        for text in (b'Research checkpoint projection requires INT4 G128 profile',b'Stage capture requires embedded mixed27B layer3 B1/2 T16 profile'):
            self.assertIn(text,old[p['modified_source']]);self.assertIn(text,new[p['modified_source']])
        record=json.loads((M/'evidence/fa3-projection-witness-build-derivation.json').read_bytes());self.assertEqual(len(record['files']),3)
        for row in record['files']:
            raw=(M/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for c in row['changes']:
                a=c['before'].encode();self.assertEqual(raw.count(a),c['count']);raw=raw.replace(a,c['after'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
    def test_wrong_norm_projection_weight_or_output_layout_cannot_pass_composition(self):
        old,new,p=sources();name=p['modified_source']
        for a,b in [(b'layer+".input_layernorm.weight",1.e-6f',b'layer+".input_layernorm.weight",1.e-5f'),(b'found->second.rows!=1024',b'found->second.rows!=5120'),(b'.self_attn.k_proj.weight',b'.self_attn.q_proj.weight'),(b'dimensions[1]*1024',b'dimensions[1]*1023')]:
            bad=dict(new);self.assertIn(a,bad[name]);bad[name]=bad[name].replace(a,b)
            with self.assertRaises(AssertionError):validate(old,bad,p)
    def test_witness_has_explicit_optin_and_never_replaces_existing_projection_route(self):
        _,new,p=sources();raw=new[p['modified_source']];self.assertEqual(raw.count(b'opName=="qwen35_fp8_fa3_projection_witness"'),1)
        for text in (b'GARNET_FP8_FA3_PROJECTION_WITNESS',b'!external.empty()',b'source->getType()==DataType::kFLOAT',b'found->second.logicalName!=name',b'ops.project(normalized,weight)'):self.assertIn(text,raw)
if __name__=='__main__':unittest.main()
