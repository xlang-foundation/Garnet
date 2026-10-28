from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from .exact_zip_payload_delta import build,apply,digest
class ZipTransport(unittest.TestCase):
    def fixture(self,root):
        a=root/'parent.zip';b=root/'original.zip';data=b''.join(hashlib.sha256(str(i).encode()).digest() for i in range(4096))
        # Different paths, timestamps, order and new metadata must survive exactly.
        with zipfile.ZipFile(a,'x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('old/input.f32',data);z.writestr('old/meta.json','old')
        with zipfile.ZipFile(b,'x',zipfile.ZIP_DEFLATED) as z:
            info=zipfile.ZipInfo('new/input.f32',(2026,10,26,10,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data);z.writestr('new/meta.json','changed');z.comment=b'preserve central directory and comment'
        return a,b
    def test_exact_original_archive_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=self.fixture(root);delta=root/'delta.zip';q=build(a,b,delta,digest(b));out=root/'reconstructed'/'original.zip';out.parent.mkdir()
            apply(a,delta,out,digest(b));self.assertEqual(out.read_bytes(),b.read_bytes())
            self.assertGreater(q['reused_payload_bytes'],0)
    def test_wrong_parent_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=self.fixture(root);delta=root/'delta.zip';build(a,b,delta,digest(b));a.write_bytes(b'wrong source')
            with self.assertRaises(AssertionError):apply(a,delta,root/'x'/'original.zip',digest(b))
    def test_segment_or_inline_mutation_rejected(self):
        for kind in ('inline','gap'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);a,b=self.fixture(root);delta=root/'delta.zip';build(a,b,delta,digest(b))
                with zipfile.ZipFile(delta) as z:m=json.loads(z.read('delta-manifest.json'));raw=z.read('inline.bin')
                if kind=='inline':raw=bytes([raw[0]^1])+raw[1:]
                else:m['segments'][0]['target_offset']=1
                bad=root/'bad.zip'
                with zipfile.ZipFile(bad,'x',zipfile.ZIP_DEFLATED) as z:z.writestr('delta-manifest.json',json.dumps(m));z.writestr('inline.bin',raw)
                dest=root/'result'/'original.zip';dest.parent.mkdir()
                with self.assertRaises(AssertionError):apply(a,bad,dest,digest(b))
                self.assertFalse(dest.exists())
if __name__=='__main__':unittest.main()
