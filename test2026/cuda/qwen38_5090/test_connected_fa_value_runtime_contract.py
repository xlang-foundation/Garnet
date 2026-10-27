from pathlib import Path
import ast,json,unittest
from .connected_fa_value_runtime import open_component_io
from .fa_value_sdk_control import verify_derivations
class RuntimeContract(unittest.TestCase):
    def test_component_IO_needs_no_embedding_head_and_closes_failed_open(self):
        class IO:
            def __init__(self,fail=False):self.calls=[];self.fail=fail
            def open(self,path):self.calls.append(('open',path));assert not self.fail,'injected open failure'
            def close(self):self.calls.append(('close',))
        class API:
            def __init__(self,value):self.value=value
            def cuda_tensor_io(self):return self.value
        value=IO();self.assertIs(open_component_io(API(value),'checkpoint'),value);self.assertEqual(value.calls,[('open','checkpoint')]);value=IO(True)
        with self.assertRaises(AssertionError):open_component_io(API(value),'checkpoint')
        self.assertEqual(value.calls,[('open','checkpoint'),('close',)]);m=Path(__file__).resolve().parent;verify_derivations(json.loads((m/'evidence/connected-fa-value-v3-derivation.json').read_bytes()),lambda name:(m/name).read_bytes());tree=ast.parse((m/'connected_fa_value_runtime.py').read_bytes());self.assertFalse(any(isinstance(node,(ast.Import,ast.ImportFrom)) for node in ast.walk(tree)))
if __name__=='__main__':unittest.main()
