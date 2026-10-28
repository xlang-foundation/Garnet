import hashlib,io,unittest,zipfile
from .b4_http_source_bundle import load_bundle,TRANSPORT

class HTTPSourceBundleControls(unittest.TestCase):
    def fixture(self):
        values={'src/file'+str(i)+'.cpp':str(i).encode() for i in range(89)}
        model={n:hashlib.sha256(b).hexdigest() for n,b in values.items()};values.update({n:n.encode() for n in TRANSPORT})
        pins={n:hashlib.sha256(b).hexdigest() for n,b in values.items()}
        return values,model,pins
    def archive(self,values):
        out=io.BytesIO()
        with zipfile.ZipFile(out,'w') as z:
            for n,b in values.items():z.writestr(n,b)
        out.seek(0);return out
    def test_exact89_plus_two_transport_members_are_admitted(self):
        values,model,pins=self.fixture();self.assertEqual(load_bundle(self.archive(values),model,pins),values)
    def test_unapproved_test_path_or_model_hash_change_is_rejected(self):
        values,model,pins=self.fixture();bad=dict(pins);bad['test2026/unapproved.py']=bad.pop(next(iter(TRANSPORT)))
        with self.assertRaises(AssertionError):load_bundle(self.archive(values),model,bad)
        bad=dict(pins);bad[next(iter(model))]='0'*64
        with self.assertRaises(AssertionError):load_bundle(self.archive(values),model,bad)
    def test_missing_extra_drifted_and_traversal_members_are_rejected(self):
        values,model,pins=self.fixture()
        for mutation in ('missing','extra','bytes','traversal'):
            changed=dict(values);expected=dict(pins);qualified=dict(model)
            if mutation=='missing':changed.pop(next(iter(values)))
            elif mutation=='extra':changed['src/extra.cpp']=b'x'
            elif mutation=='bytes':changed[next(iter(values))]=b'drift'
            else:
                old=next(iter(model));raw=changed.pop(old);h=expected.pop(old);qualified.pop(old);bad='src/../escape.cpp';changed[bad]=raw;expected[bad]=qualified[bad]=h
            with self.assertRaises(AssertionError):load_bundle(self.archive(changed),qualified,expected)
