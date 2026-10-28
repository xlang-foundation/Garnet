"""CPU corruption controls: source/runtime restoration and unchanged HTTP logic."""
import ast,unittest
from pathlib import Path
from .native_o128_http_admission import validate_closed,LIBRARY,BUNDLE
M=Path(__file__).resolve().parent

class Certificate(unittest.TestCase):
    def fixture(self):
        candidate={'file'+str(i):str(i) for i in range(60)}
        p=dict(files={'base':'basehash'},candidate_files=candidate,installed_files={'base':'basehash',**{'Garnet/'+n:h for n,h in candidate.items()}},candidate_library_sha256=LIBRARY,source_bundle_sha256=BUNDLE,runtime={'garnet.dll':'baseline'},environment={'sdk':'sdkhash'},checkpoints={'config':'confighash'})
        w=dict(exit_code=0,cleanup_finished=True,baseline_source_restored=True,baseline_library_restored=True,source_before_sha256=p['files'],source_after_sha256=p['files'],installed_source_sha256=p['installed_files'],candidate_library_sha256=LIBRARY,runtime_dependencies=p['runtime'],runtime_dependencies_after=p['runtime'],environment_before=p['environment'],environment_after=p['environment'],checkpoint_after=p['checkpoints'],performance_admitted=False)
        return p,w
    def test_complete_closed_certificate(self):validate_closed(*self.fixture())
    def test_rejects_failed_unrestored_and_unqualified_records(self):
        for key,bad in [('exit_code',True),('exit_code',1),('cleanup_finished',False),('baseline_source_restored',False),('baseline_library_restored',False),('candidate_library_sha256','other'),('performance_admitted',True)]:
            p,w=self.fixture();w[key]=bad
            with self.subTest(key=key,bad=bad),self.assertRaises(AssertionError):validate_closed(p,w)
    def test_rejects_source_runtime_sdk_checkpoint_drift(self):
        for key in ('source_after_sha256','installed_source_sha256','runtime_dependencies_after','environment_after','checkpoint_after'):
            p,w=self.fixture();w[key]={}
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_closed(p,w)
        for key in ('candidate_files','source_bundle_sha256'):
            p,w=self.fixture();p[key]={} if key=='candidate_files' else 'wrong'
            with self.subTest(key=key),self.assertRaises(AssertionError):validate_closed(p,w)
    def test_wire_and_callback_function_AST_unchanged(self):
        def functions(name):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse((M/name).read_bytes())) if isinstance(n,ast.FunctionDef)}
        self.assertEqual(functions('native_o128_stream_worker.py'),functions('native_prefill16_stream_worker.py'))
if __name__=='__main__':unittest.main()
