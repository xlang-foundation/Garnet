"""CPU-only raw-entry, immutable parent and exact lifecycle/IPC derivation faults."""
from pathlib import Path
import copy,hashlib,json,unittest
from .native_cuda_phase_entry import instrument
from .verify_native_cuda_entry_profiled import replay_entry
from .prepare_native_cuda_entry_profiled import admit_parent

M=Path(__file__).resolve().parent;sha=lambda raw:hashlib.sha256(raw).hexdigest()

class Controls(unittest.TestCase):
    def test_parent_requires_actual_terminal_unprofiled_ordinary_control(self):
        baseline='1a1d0229a1c0c24a8397a4dee7a614bcedf97b3bd3293e58d752ed7de30eb498'
        q={'status':'native_cuda_logged_bootstrap_replayed','archive_sha256':'a'*64,'profiled':False,'performance_admitted':False,
           'two_times_target_achieved':False,'library_sha256':baseline,
           'control_replay':{'requests':[{'mode':'ordinary','tokens_bound_to_retained_4k_control':True}]}}
        w={'exit_code':0,'cleanup_finished':True,'library_unchanged':True,'library_sha256':baseline,
           'source_sha256':{'source':'b'},'source_after_sha256':{'source':'b'},
           'runtime_dependencies':{'garnet.dll':baseline},'runtime_dependencies_after':{'garnet.dll':baseline}}
        admit_parent(q,w,'a'*64)
        for fault in ('exit','bool_exit','cleanup','source','runtime','tokens','profiled','missing_request','archive'):
            p,a=copy.deepcopy(q),copy.deepcopy(w)
            if fault=='exit':a['exit_code']=1
            elif fault=='bool_exit':a['exit_code']=False
            elif fault=='cleanup':a['cleanup_finished']=False
            elif fault=='source':a['source_after_sha256']['source']='bad'
            elif fault=='runtime':a['runtime_dependencies_after']['garnet.dll']='bad'
            elif fault=='tokens':p['control_replay']['requests'][0]['tokens_bound_to_retained_4k_control']=False
            elif fault=='profiled':p['profiled']=True
            elif fault=='missing_request':p['control_replay']['requests']=[]
            else:p['archive_sha256']='bad'
            with self.subTest(fault=fault),self.assertRaises(AssertionError):admit_parent(p,a,'a'*64)

    def test_all_raw_entry_stages_and_exact_retained_instrumented_source_are_required(self):
        original=(M/'native_cuda_phase_logged_worker.py').read_bytes();source=instrument(original.decode()).encode()
        stages=['python_entry_started','instrumented_source_retained','before_garnet_import','garnet_import_returned',
                'before_model_import','model_import_returned','worker_execution_returned']
        rows=[{'ordinal':i,'stage':s,'mode':'ordinary','monotonic_seconds':float(i)} for i,s in enumerate(stages)]
        rows[0]['pid']=123;rows[1].update(original_sha256=sha(original),instrumented_sha256=sha(source))
        self.assertEqual(replay_entry(rows,source,original)['events'],7)
        for fault in ('missing','stage','source','hash','clock','mode','bool_ordinal'):
            bad=copy.deepcopy(rows);raw=source
            if fault=='missing':bad.pop(3)
            elif fault=='stage':bad[3]['stage']='worker_raised'
            elif fault=='source':raw=source+b'\nprint("unqualified")\n'
            elif fault=='hash':bad[1]['original_sha256']='bad'
            elif fault=='clock':bad[4]['monotonic_seconds']=-1.
            elif fault=='mode':bad[0]['mode']='mtp'
            else:bad[0]['ordinal']=False
            with self.subTest(fault=fault),self.assertRaises(AssertionError):replay_entry(bad,raw,original)

    def test_exact_derivation_keeps_original_request_and_cleanup_operations(self):
        r=json.loads((M/'evidence/native-cuda-entry-profiled-derivation.json').read_bytes())
        for name,row in r['files'].items():
            source=row['source_text'];self.assertEqual(sha(source.encode()),row['source_sha256'])
            for step in row['replacements']:
                self.assertEqual(source.count(step['before']),step['count']);source=source.replace(step['before'],step['after'])
            self.assertEqual(source.encode(),(M/name).read_bytes());self.assertEqual(sha(source.encode()),row['target_sha256'])
        collector=(M/'collect_native_cuda_entry_profiled.py').read_text()
        for flag in ('--trace=cuda','--cuda-graph-trace=node','--force-overwrite=false','replay(connection,markers,host)'):
            self.assertIn(flag,collector)
        self.assertNotIn('synthetic',collector)

if __name__=='__main__':unittest.main()
