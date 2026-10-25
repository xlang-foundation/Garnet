"""CPU admission faults and literal unchanged generation/protocol bodies."""
from pathlib import Path
import ast,copy,os,unittest
from unittest.mock import patch
from .native_tiled_scores_http_admission import BASELINE,CANDIDATE,MODEL_ARCHIVE,validate_model,require
M=Path(__file__).resolve().parent

def fixture():
    routes={}
    for mode in ('default','tiled'):
        routes[mode]={'tiled_scores_enabled':mode=='tiled','captured_states':32,'complete_retained_baseline_states_and_raw_vectors_exact':True,
                      'cases':[{'actual_prompt_tokens':3968,'actual_output_tokens':128,'target_complete_state_and_raw_vectors_exact':True} for _ in range(6)]}
    p={'candidate_library_sha256':CANDIDATE,'baseline_library_sha256':BASELINE,'status':'native_tiled_scores_model_replayed','archive_sha256':MODEL_ARCHIVE,
       'baseline_source_and_runtime_restored':True,'performance_admitted':False,'two_times_target_achieved':False,'routes':routes}
    w={'candidate_library_sha256':CANDIDATE,'baseline_library_sha256':BASELINE,'exit_code':0,'cleanup_finished':True,'baseline_library_restored':True,
       'source_sha256':{'synthetic':'a'},'source_after_sha256':{'synthetic':'a'},'runtime_dependencies':{'garnet.dll':BASELINE},'runtime_dependencies_after':{'garnet.dll':BASELINE},'environment_before':{'synthetic':'b'},'environment_after':{'synthetic':'b'}}
    return p,w

class TiledHTTPControls(unittest.TestCase):
    def test_both_complete_routes_required_before_any_http_promotion(self):
        p,w=fixture();validate_model(p,w,CANDIDATE)
        for kind in ('library','live','source','sdk','missing_route','wrong_route','state_count','short_context','vector_not_exact','performance_promotion'):
            p,w=fixture();library=CANDIDATE
            if kind=='library':library=BASELINE
            elif kind=='live':w['cleanup_finished']=False
            elif kind=='source':w['source_after_sha256']={}
            elif kind=='sdk':w['environment_after']={}
            elif kind=='missing_route':del p['routes']['default']
            elif kind=='wrong_route':p['routes']['tiled']['tiled_scores_enabled']=False
            elif kind=='state_count':p['routes']['tiled']['captured_states']=31
            elif kind=='short_context':p['routes']['tiled']['cases'][0]['actual_prompt_tokens']=41
            elif kind=='vector_not_exact':p['routes']['default']['complete_retained_baseline_states_and_raw_vectors_exact']=False
            else:p['performance_admitted']=True
            with self.subTest(kind=kind),self.assertRaises(AssertionError):validate_model(p,w,library)
    def test_wrong_mode_or_route_flag_rejected_before_reading_files(self):
        with self.assertRaises(ValueError):require(Path('nonexistent-synthetic-root'),'ordinary',CANDIDATE)
        for value in ('','0','true','01'):
            with patch.dict(os.environ,{'GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES':value}),self.assertRaises(ValueError):require(Path('nonexistent-synthetic-root'),'mtp',CANDIDATE)
    def test_entire_generation_and_request_body_is_literal_unchanged(self):
        original=(M/'native_prefill16_stream_worker.py').read_text();candidate=(M/'native_tiled_scores_stream_worker.py').read_text()
        start="checkpoint=str(root/'models/Qwen3.8-27B-NVFP4')"
        self.assertEqual(original[original.index(start):],candidate[candidate.index(start):])
        function=lambda s:next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='validate_body')
        self.assertEqual(ast.dump(function(original),include_attributes=False),ast.dump(function(candidate),include_attributes=False))
        self.assertIn('assert garnet.bf16_low_batch_tiled_scores_enabled() is True',candidate)
    def test_entire_http_probe_ast_only_changes_baseline_library_checks_and_worker_names(self):
        original=(M/'probe_prefill16_http.py').read_text();candidate=(M/'probe_tiled_scores_http.py').read_text()
        candidate=candidate.replace('serve_native_tiled_scores_worker.bat','serve_native_prefill16_worker.bat').replace('native_tiled_scores_stream_worker.py','native_prefill16_stream_worker.py')
        candidate=candidate.replace("proof['library_sha256']=='"+BASELINE+"'","proof['library_sha256']==library").replace("ordinary_report['library_sha256']=='"+BASELINE+"'","ordinary_report['library_sha256']==library")
        self.assertEqual(ast.dump(ast.parse(original),include_attributes=False),ast.dump(ast.parse(candidate),include_attributes=False))
    def test_launcher_only_selects_tiled_flag_and_worker_entry(self):
        original=(M/'serve_native_prefill16_worker.bat').read_bytes();candidate=(M/'serve_native_tiled_scores_worker.bat').read_bytes()
        newline=b'\r\n' if b'\r\n' in candidate else b'\n'
        self.assertEqual(candidate.replace(b'set GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES=1'+newline,b'').replace(b'native_tiled_scores_stream_worker.py',b'native_prefill16_stream_worker.py'),original)

if __name__=='__main__':unittest.main()
