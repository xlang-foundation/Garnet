import copy,unittest
from .fp8_decode_shape_configs import replay,GEOMETRIES,KNOWN
UUID='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
class Configs(unittest.TestCase):
    def fixture(self):
        q=dict(status='actual_live_FP8_M4_M16_geometry_configurations_collected_no_matmul',device=0,compute_capability=[12,0],cublasLt_version=130400,uuid_hex=UUID[4:].replace('-',''),runtime_version=13020,CUDART_VERSION=13020,driver_version=13020,cublasLt_DLL='D:/CantorAI/5090-study-20261023/tools/cuda13/nvidia/cu13/bin/x86_64/cublasLt64_13.dll',weight_dtype='E4M3',activation_dtype='E4M3',output_dtype='BF16',compute_dtype='FP32',TRANSA='T',TRANSB='N',matmul_executed=False,model_loaded=False,performance_qualified=False,geometries=[],API_events=[])
        def event(name,count=1):q['API_events'].extend(dict(name=name,status=0) for _ in range(count))
        for name in ('cudaGetDevice','cudaGetDeviceProperties','cudaDriverGetVersion','cudaRuntimeGetVersion','cublasLtCreate','cublasLtDestroy'):event(name)
        for label,n,k in GEOMETRIES:
            for name in ('cublasLtMatmulDescCreate','cublasLtMatmulDescDestroy','cublasLtMatmulDescSetAttribute:TRANSA','cublasLtMatmulDescSetAttribute:TRANSB','cublasLtMatmulPreferenceCreate','cublasLtMatmulPreferenceDestroy','cublasLtMatmulPreferenceSetAttribute'):event(name)
            for name in ('W','X4','Y4','X16','Y16'):event('cublasLtMatrixLayoutCreate:'+name)
            event('cublasLtMatrixLayoutDestroy',5)
            for m in (4,16):
                c=list(KNOWN) if m==4 else [35,15,18,7,2,1,0];q['geometries'].append(dict(label=label,M=m,N=n,K=k,workspace_limit=4<<20,found=1,choices=[dict(index=0,state=0,heuristic_workspace=100,opaque='ab'*64,configuration=c,checks=[dict(M=x,status=0,state=0,workspace=100) for x in (4,16)])]));event('cublasLtMatmulAlgoGetHeuristic');event('cublasLtMatmulAlgoConfigGetAttribute',7);event('cublasLtMatmulAlgoCheck',2)
        return q
    def test_full_configuration_eligibility_never_promotes_numerical_result(self):
        r=replay(self.fixture(),UUID);self.assertEqual(r['actual_AlgoChecks'],20);self.assertTrue(r['all_known_M4_references_match']);self.assertFalse(r['rows'][2]['M4_and_M16_configurations_equal']);self.assertFalse(r['numerical_or_model_or_performance_qualified'])
    def test_wrong_identity_geometry_missing_attributes_and_API_errors_rejected(self):
        for change in ('uuid','geometry','attributes','API','count'):
            q=self.fixture()
            if change=='uuid':q['uuid_hex']='0'*32
            elif change=='geometry':q['geometries'][6]['K']=6144
            elif change=='attributes':q['geometries'][0]['choices'][0]['configuration'].pop()
            elif change=='API':q['API_events'][0]['status']=7
            else:q['API_events'].pop()
            with self.assertRaises(AssertionError):replay(q,UUID)
    def test_workspace_and_AlgoCheck_failure_are_observations_not_admission(self):
        q=self.fixture();q['geometries'][6]['choices'][0]['checks'][1]['workspace']=(4<<20)+1;r=replay(q,UUID);self.assertFalse(r['rows'][3]['live_M4_eligible_at_M4_and_M16'])
        q=self.fixture();q['geometries'][6]['choices'][0]['checks'][1]['status']=7;events=[r for r in q['API_events'] if r['name']=='cublasLtMatmulAlgoCheck'];events[13]['status']=7;r=replay(q,UUID);self.assertFalse(r['rows'][3]['live_M4_eligible_at_M4_and_M16']);self.assertFalse(r['numerical_or_model_or_performance_qualified'])
if __name__=='__main__':unittest.main()
