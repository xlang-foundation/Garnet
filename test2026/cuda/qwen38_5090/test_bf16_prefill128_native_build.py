"""CPU fault controls for the real native source transaction; no runtime or GPU."""
from pathlib import Path
import tempfile,unittest,zipfile
from .run_bf16_prefill128_native_build import candidate_bytes,snapshot_targets,restore_targets,digest
from .verify_bf16_prefill128_native_build import wire_contract

class Transactions(unittest.TestCase):
    def test_wire_replay_requires_every_count_and_no_enqueue(self):
        expected='D:/owned/nvinfer_10.dll'
        report={'status':'prefill128_plugin_cpu_contract_passed','profiles':16,'roundtrips':16,'clones':16,'bad_wires':832,'bad_descriptors':496,'bad_formats':336,'cross_dialect_rejections':128,'legacy_roundtrips':96,'unexpected_operator_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'runtime_version':100000,'header_version':100000,'loaded_tensorrt_dll':expected}
        self.assertEqual(wire_contract(report,expected)['profiles'],16)
        for name in ('profiles','roundtrips','clones','bad_wires','bad_descriptors','bad_formats','cross_dialect_rejections','legacy_roundtrips','unexpected_operator_calls'):
            with self.assertRaises(AssertionError):wire_contract(dict(report,**{name:report[name]+1}),expected)
        for name in ('enqueue_tested','model_qualified','performance_admitted'):
            with self.assertRaises(AssertionError):wire_contract(dict(report,**{name:True}),expected)

    def test_wire_replay_rejects_foreign_runtime_or_boolean_count(self):
        report={'status':'prefill128_plugin_cpu_contract_passed','profiles':16,'roundtrips':16,'clones':16,'bad_wires':832,'bad_descriptors':496,'bad_formats':336,'cross_dialect_rejections':128,'legacy_roundtrips':96,'unexpected_operator_calls':0,'enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'runtime_version':100000,'header_version':100000,'loaded_tensorrt_dll':'D:/owned/nvinfer_10.dll'}
        for changes in ({'loaded_tensorrt_dll':'D:/outside/nvinfer_10.dll'},{'header_version':99999},{'profiles':True}):
            with self.assertRaises(AssertionError):wire_contract(dict(report,**changes),'D:/owned/nvinfer_10.dll')

    def test_existing_and_new_sources_restore_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder);(repo/'src').mkdir();old=b'original\r\n';new=b'candidate\n';(repo/'src/existing.cpp').write_bytes(old)
            candidates={'src/existing.cpp':new,'src/added.cpp':new};base={'src/existing.cpp':digest(old),'src/added.cpp':None}
            previous=snapshot_targets(repo,candidates,base)
            for name,raw in candidates.items():(repo/name).write_bytes(raw)
            restore_targets(repo,candidates,previous)
            self.assertEqual((repo/'src/existing.cpp').read_bytes(),old);self.assertFalse((repo/'src/added.cpp').exists())

    def test_outside_edit_blocks_all_restore_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder);(repo/'src').mkdir();candidate={'src/a.cpp':b'a','src/b.cpp':b'b'}
            for name,raw in candidate.items():(repo/name).write_bytes(raw)
            (repo/'src/b.cpp').write_bytes(b'external')
            with self.assertRaises(AssertionError):restore_targets(repo,candidate,{'src/a.cpp':b'old-a'})
            self.assertEqual((repo/'src/a.cpp').read_bytes(),b'a');self.assertEqual((repo/'src/b.cpp').read_bytes(),b'external')

    def test_partial_install_restores_only_written_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder);(repo/'src').mkdir();old={'src/a.cpp':b'old-a','src/b.cpp':b'old-b'}
            for name,raw in old.items():(repo/name).write_bytes(raw)
            (repo/'src/a.cpp').write_bytes(b'candidate')
            restore_targets(repo,{'src/a.cpp':b'candidate'},old)
            self.assertEqual({name:(repo/name).read_bytes() for name in old},old)

    def test_unrecorded_target_and_wrong_base_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder);(repo/'src').mkdir();(repo/'src/a.cpp').write_bytes(b'outside');candidate={'src/a.cpp':b'new'}
            for base in ({'src/a.cpp':None},{'src/a.cpp':digest(b'wrong')}):
                with self.assertRaises(AssertionError):snapshot_targets(repo,candidate,base)
            self.assertEqual((repo/'src/a.cpp').read_bytes(),b'outside')

    def test_bundle_digest_member_set_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'source.zip'
            for name,raw,expected in (('src/a.cpp',b'ok',{'src/a.cpp':digest(b'ok')}),('src/a.cpp',b'bad',{'src/a.cpp':digest(b'ok')}),('src/../outside',b'ok',{'src/../outside':digest(b'ok')}),('src/extra.cpp',b'ok',{'src/a.cpp':digest(b'ok')})):
                with zipfile.ZipFile(path,'w') as z:z.writestr(name,raw)
                if name=='src/a.cpp' and raw==b'ok':self.assertEqual(candidate_bytes(path,expected),{name:raw})
                else:
                    with self.assertRaises(AssertionError):candidate_bytes(path,expected)

if __name__=='__main__':unittest.main()
