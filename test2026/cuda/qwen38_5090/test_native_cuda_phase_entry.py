"""CPU-only import-failure/exit/IPC controls; actual worker is never executed."""
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import ast,builtins,contextlib,io,json,tempfile,unittest
from .native_cuda_phase_entry import EntryLog,execute,instrument,HOOK,IMPORTS

M=Path(__file__).resolve().parent
FIXTURE='\n'+IMPORTS[0][0]+'\n'+IMPORTS[1][0]+'\nprint("synthetic-output")\n'

class Controls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='cpu-entry-fixture-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def run_fixture(self,source=FIXTURE,import_error=None):
        log=EntryLog(self.root/'entry.jsonl','ordinary');output=io.StringIO();real_import=builtins.__import__
        def fake_import(name,*args,**kwargs):
            if name=='garnet':
                if import_error is not None:raise import_error
                return SimpleNamespace()
            if name=='xModel.qwen38.vl_27b.model':
                return SimpleNamespace(create_cuda_bf16_low_batch_study_decoder=None,create_cuda_vision=None,prepare_cuda_image=None)
            return real_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=fake_import),contextlib.redirect_stdout(output):
            execute(source,M/'native_cuda_phase_logged_worker.py',self.root/'source.py.txt',log)
        return output.getvalue()

    def rows(self):return [json.loads(v) for v in (self.root/'entry.jsonl').read_bytes().splitlines()]

    def test_success_preserves_stdout_and_retains_each_native_import_milestone(self):
        self.assertEqual(self.run_fixture(),'synthetic-output\n')
        rows=self.rows();self.assertEqual([r['ordinal'] for r in rows],list(range(7)))
        self.assertEqual([r['stage'] for r in rows],['python_entry_started','instrumented_source_retained',
            'before_garnet_import','garnet_import_returned','before_model_import','model_import_returned','worker_execution_returned'])
        self.assertEqual((self.root/'source.py.txt').read_text(),instrument(FIXTURE))

    def test_original_native_import_failure_is_logged_and_reraised(self):
        error=ImportError('synthetic native bootstrap failure')
        with self.assertRaises(ImportError) as observed:self.run_fixture(import_error=error)
        self.assertIs(observed.exception,error);rows=self.rows()
        self.assertEqual([r['stage'] for r in rows][-2:],['before_garnet_import','worker_raised'])
        self.assertIn('synthetic native bootstrap failure',rows[-1]['exception']['traceback'])

    def test_system_exit_retains_original_code_and_log_cannot_be_overwritten(self):
        with self.assertRaises(SystemExit) as error:self.run_fixture(FIXTURE+'raise SystemExit(7)\n')
        self.assertEqual(error.exception.code,7);self.assertEqual(self.rows()[-1]['exception']['type'],'SystemExit')
        original=(self.root/'entry.jsonl').read_bytes()
        with self.assertRaises(FileExistsError):EntryLog(self.root/'entry.jsonl','ordinary')
        self.assertEqual((self.root/'entry.jsonl').read_bytes(),original)

    def test_instrumentation_cannot_change_qualified_model_operations(self):
        source=(M/'native_cuda_phase_logged_worker.py').read_bytes().decode();original=ast.parse(source)
        candidate=ast.parse(instrument(source))
        candidate.body=[n for n in candidate.body if not(isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
            isinstance(n.value.func,ast.Name) and n.value.func.id==HOOK)]
        self.assertEqual(ast.dump(candidate,include_attributes=False),ast.dump(original,include_attributes=False))
        for bad in (source.replace('import garnet','import something_else'),source+'\n'+HOOK+'=None\n',source+'\nimport garnet\n'):
            with self.assertRaises(ValueError):instrument(bad)

    def test_launcher_retains_original_exit_and_identifies_failure_before_python(self):
        source=(M/'serve_native_cuda_phase_entry_worker.bat').read_bytes()
        self.assertNotIn(b'\r',source)
        self.assertIn(b'native_cuda_phase_entry.py',source)
        self.assertIn(b'set "GARNET_DIAGNOSTIC_ENTRY_EXIT=%errorlevel%"',source)
        self.assertTrue(source.endswith(b'exit /b %GARNET_DIAGNOSTIC_ENTRY_EXIT%\n'))
        self.assertIn(b'GARNET_DIAGNOSTIC_LAUNCHER_ENTERED 1>&2',source)

if __name__=='__main__':unittest.main()
