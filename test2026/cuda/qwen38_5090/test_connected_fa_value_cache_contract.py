from pathlib import Path
import ast,hashlib,json,unittest
from .fa_value_sdk_control import verify_derivations
class Buffer:
    def __init__(self,shape):self.shape=shape
class IO:
    def storage_view(self,source,shape):
        assert len(source.shape)==4 and shape==[source.shape[0],4194304];return Buffer(shape)
    def slice_into(self,source,start,target):
        assert len(source.shape)==2 and source.shape[0]==target.shape[0] and 0<=start<=source.shape[1]-target.shape[1]
    def row(self,part,row):assert 0<=row<part.shape[0];return Buffer([1,part.shape[1]])
    def finite_f32_bytes(self,part):return bytes(part.shape[0]*part.shape[1]*4)
class CacheContract(unittest.TestCase):
    def test_actual_complete_cache_reader_requires_packed_view_and_covers_all_rows(self):
        m=Path(__file__).resolve().parent;verify_derivations(json.loads((m/'evidence/connected-fa-value-v4-derivation.json').read_bytes()),lambda name:(m/name).read_bytes())
        for batch in (1,4):
            for version in (3,4):
                tree=ast.parse((m/('probe_connected_fa_value_v'+str(version)+'.py')).read_bytes());nodes=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='cache_hashes'];self.assertEqual(len(nodes),1);scope=dict(io=IO(),hashlib=hashlib,batch=batch,width=4194304,cachepart=Buffer([batch,1048576]));module=ast.Module(body=nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(module),'actual_cache_reader','exec'),scope);caches=[Buffer([batch,4,4096,256]) for _ in range(2)]
                if version==3:
                    with self.assertRaises(AssertionError):scope['cache_hashes'](caches)
                else:self.assertEqual(scope['cache_hashes'](caches),[hashlib.sha256(bytes(4194304*4)).hexdigest()]*(2*batch))
if __name__=='__main__':unittest.main()
