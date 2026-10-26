"""Synthetic metadata/state/source faults only; no model or GPU measurements."""
from pathlib import Path
import ast,copy,hashlib,json,unittest
from .test_mtp_prefill16_full4k import fixture
from .prefill128_model import replay_prefill128_model

def data():
    reference,files,cases=fixture();actual=copy.deepcopy(reference)
    actual.update(target128_enabled=True,target_prefill_chunk=128,target_decode_chunk=4,draft_prefill_chunk=16,target_program_chunks=[4,128],tiled_scores_enabled=False,
      prefill_helper={'ready':True,'poisoned':False,'batch':1,'chunk':128,'canonical_state_width':106823680,'linear_replay_width':39714816,'target_width':44564480},
      decode_helper={'ready':True,'poisoned':False,'batch':1,'chunk':4,'canonical_state_width':106823680,'linear_replay_width':39714816},
      memory_after_construction={'target_prefill_chunk':128,'draft_prefill_chunk':16,'draft_decode_chunk':4,'study_gpu_qualified':False,'study_performance_admitted':False})
    keys=('sequence','case_index','key','request_sha256','image_sha256','prompt_ids','ordinary_ids','reset_state','ordinary_prefill','ordinary_final')
    actual['ordinary_observations']=[{key:copy.deepcopy(row[key]) for key in keys} for row in actual['cases']]
    return actual,reference,files,cases

class Controls(unittest.TestCase):
    def test_six_complete_cases_keep_exact_retained_oracle(self):
        actual,reference,files,cases=data();result=replay_prefill128_model(actual,files.__getitem__,reference,files.__getitem__,cases)
        self.assertEqual(result['captured_states'],32);self.assertFalse(result['performance_admitted']);self.assertEqual(result['target_prefill_chunk'],128)

    def test_profile_helper_and_quality_promotion_faults_fail(self):
        for fault in ('old_chunk','bool_chunk','missing_optin','programs','helper_chunk','helper_target','helper_poison','memory','quality'):
            a,r,f,c=data()
            if fault=='old_chunk':a['target_prefill_chunk']=32
            elif fault=='bool_chunk':a['target_decode_chunk']=True
            elif fault=='missing_optin':a['target128_enabled']=False
            elif fault=='programs':a['target_program_chunks']=[4,32]
            elif fault=='helper_chunk':a['prefill_helper']['chunk']=32
            elif fault=='helper_target':a['prefill_helper']['target_width']-=1
            elif fault=='helper_poison':a['decode_helper']['poisoned']=True
            elif fault=='memory':a['memory_after_construction']['draft_decode_chunk']=16
            else:a['memory_after_construction']['study_gpu_qualified']=True
            with self.subTest(fault=fault),self.assertRaises(AssertionError):replay_prefill128_model(a,f.__getitem__,r,f.__getitem__,c)

    def test_shared_ordinary_mtp_drift_still_fails_retained_state(self):
        a,r,f,c=data()
        for row in a['cases']:
            if row['case_index']==0:
                for name in ('ordinary_final','candidate_final'):row[name]['logical']['components'][0]['row_sha256']=['f'*64]
        with self.assertRaises(AssertionError):replay_prefill128_model(a,f.__getitem__,r,f.__getitem__,c)

    def test_ordinary_evidence_cannot_be_missing_or_relabelled(self):
        for fault in ('missing','ids','state'):
            a,r,f,c=data()
            if fault=='missing':a['ordinary_observations'].pop()
            elif fault=='ids':a['ordinary_observations'][0]['ordinary_ids'][0]+=1
            else:a['ordinary_observations'][0]['ordinary_prefill']['logical']['components'][-1]['row_sha256']=['b'*64]
            with self.subTest(fault=fault),self.assertRaises(AssertionError):replay_prefill128_model(a,f.__getitem__,r,f.__getitem__,c)

    def test_recorded_derivation_preserves_old_draft_algorithms_and_saves_before_assert(self):
        m=Path(__file__).resolve().parent;repo=m.parents[2];record=json.loads((m/'evidence/prefill128-model-source-derivation.json').read_bytes())
        for row in record['files']:
            raw=(repo/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            self.assertEqual(text.encode(),(repo/row['target']).read_bytes());self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),row['target_sha256'])
        roots=[]
        for name in ('cuda_mtp_prefill16_study.py','cuda_mtp_prefill16_target128_study.py'):
            tree=ast.parse((repo/'xModel/qwen35/vl_0_8b'/name).read_bytes());roots.append(next(node for node in tree.body if isinstance(node,ast.ClassDef)))
        for name in ('_prefill_hook','reset','memory_info','close'):
            methods=[next(node for node in root.body if isinstance(node,ast.FunctionDef) and node.name==name) for root in roots]
            self.assertEqual(ast.dump(methods[0]),ast.dump(methods[1]))
        source=(m/'probe_bf16_prefill128_model.py').read_text();self.assertLess(source.index("report['ordinary_observations'].append"),source.index("assert ordinary==case['generated_ids_by_mode']['ordinary']"))

if __name__=='__main__':unittest.main()
