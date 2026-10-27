import copy,unittest
from .fp8_policy_cache_production_contract import CASES,replay
def fixture():
    rows=[];current=None;counter=0;baseline=None;previous={}
    for s in CASES:
        phase=s['phase'];hit=phase=='hit';reject=phase=='reject'
        if phase=='miss':
            counter+=1;current=baseline if s['key']=='return-default-miss' else format(counter,'032x')
            if baseline is None:baseline=current
        cached={'runtime_graph.cache':dict(bytes=10,sha256='a'*64,fingerprint=current)};c=dict(graph_cache_hits=int(hit),graph_cache_misses=int(phase=='miss'),root_x_executions=int(phase=='miss'));status=dict(mode='compiled_xmodel',state='failed' if reject else ('engine_cache_loaded' if hit else 'compiled_engine_ready'),ready=not reject,forbidden_path_counters=c,error_code='invalid_graph_policy' if reject else '',error_message=s['expected_error'] or '')
        rows.append(dict(status='actual_production_cache_case_collected',spec=copy.deepcopy(s),actual_environment=dict(s['environment']),library_sha256='test-library',actual_Garnet_load_model_called=True,runtime_status=status,cache_before=copy.deepcopy(previous),cache_after=copy.deepcopy(cached),output_values=None if reject else [1.,6.,15.,28.],forward_status=None if reject else 'ok'));previous=cached
    return dict(status='actual_production_cache_contracts_collected',fresh_process_per_case=True,cases=rows,model_quality_or_performance_admitted=False)
class CacheProductionControls(unittest.TestCase):
    def test_full_hit_miss_alias_and_rejection_matrix(self):
        r=fixture();result=replay(r,'test-library');self.assertEqual(len(CASES),41);self.assertEqual(result['distinct_fingerprints'],13);self.assertEqual(sum(c['phase']=='miss' for c in CASES),14);self.assertEqual(sum(c['phase']=='hit' for c in CASES),16)
    def test_rejected_policy_cannot_reuse_or_modify_existing_cache(self):
        for i,s in enumerate(CASES):
            if s['phase']!='reject':continue
            for corrupt in ('hit','cache','error'):
                r=fixture();v=r['cases'][i]
                if corrupt=='hit':v['runtime_status']['forbidden_path_counters']['graph_cache_hits']=1
                elif corrupt=='cache':v['cache_after']['runtime_graph.cache']['sha256']='b'*64
                else:v['runtime_status']['error_message']='unrelated exception'
                with self.assertRaises(AssertionError):replay(r,'test-library')
    def test_policy_changes_cannot_alias_and_gpu_output_must_match(self):
        r=fixture();r['cases'][4]['cache_after']['runtime_graph.cache']['fingerprint']=r['cases'][0]['cache_after']['runtime_graph.cache']['fingerprint']
        with self.assertRaises(AssertionError):replay(r,'test-library')
        r=fixture();r['cases'][0]['output_values'][0]=0.
        with self.assertRaises(AssertionError):replay(r,'test-library')
