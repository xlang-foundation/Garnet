"""Synthetic CPU provenance/variant/budget faults; no model/server evidence."""
from pathlib import Path
import ast,copy,hashlib,io,json,unittest,zipfile
from .reference_observed_output import STABLE_METADATA,admit_observation,stream_comparison,sha
from .test_fixed_length_reference import sample
from .benchmark_fixed_length_reference import control_response


def proof_fixture(**workflow_changes):
    metadata={k:'synthetic-'+k for k in STABLE_METADATA};metadata.update(engine='sglang',generation_mode='ordinary',source_after_startup_verified=True,owned_server_pid_verified=True)
    workflow={'exit_code':0,'cleanup_finished':True,'server_exit_code':0,'source_sha256':{'s':'h'},'source_after_sha256':{'s':'h'},'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'}};workflow.update(workflow_changes);base='results/sglang-output-characterization-attempt1/';files={base+'workflow.json':json.dumps(workflow).encode(),base+'metadata.json':json.dumps(metadata).encode()};buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w') as z:
        for name,raw in files.items():z.writestr(name,raw)
        z.writestr('evidence-manifest.json',json.dumps({'files':{n:sha(v) for n,v in files.items()}}))
    raw={'status':'reference_concurrency_output_characterization_raw_replayed','actual_nonstream_id_requests':192,'actual_full4k_streams':64,'actual_color_streams':32,'concurrent_color_nonstreams':48,'concurrent_color_streams':24,'actual_client_overlap_waves':135,'serial_after_comparisons':{g:{str(i):{'completion_ids_exact':True,'text_exact':True} for i in range(8)} for g in ('text','image','mixed-color')},'physical_gpu_batch_observed':False,'cause_or_harmlessness_proven':False,'broad_quality_qualified':False,'performance_admitted':False,'two_times_target_achieved':False}
    proof={'status':'sglang_output_characterization_replayed','engine':'sglang','mode':'ordinary','archive_sha256':sha(buf.getvalue()),'replay_verifier_sha256':'a'*64,'archived_replay_verifier_sha256':'a'*64,'configured_engine_context_capacity':4098,'actual_workload_context_capacity':4096,'engine_context_reserved_slots':2,'performance_admitted':False,'two_times_target_achieved':False,'profiles':[{'raw_characterization':raw}],'files_verified':len(files)}
    return proof,buf.getvalue(),metadata


def stream_fixture():
    response,prompt=sample('sglang');control=control_response(response,'sglang',prompt);body={'ignore_eos':True,'max_tokens':128,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'messages':[]};result={'key':'synthetic','status':'completed','prompt_tokens':3968,'generated_tokens':128,'finish_reason':'length','text':control['text'],'request_sha256':sha(json.dumps(body,separators=(',',':')).encode())};return result,control,body


class ObservedOutputControls(unittest.TestCase):
    def test_closed_profile_binds_while_never_claiming_harmlessness_or_streamed_ids(self):
        p,archive,metadata=proof_fixture();r=admit_observation(json.dumps(p).encode(),archive,metadata);self.assertEqual(r['status'],'closed_observed_output_profile_bound');self.assertFalse(r['streamed_ids_observed']);self.assertFalse(r['numerical_cause_or_harmlessness_proven']);self.assertFalse(r['broad_quality_equivalence_proven'])
    def test_profile_source_context_dtype_and_cache_metadata_cannot_drift(self):
        for key in STABLE_METADATA:
            p,archive,metadata=proof_fixture();metadata[key]='changed'
            with self.subTest(key=key),self.assertRaises(AssertionError):admit_observation(json.dumps(p).encode(),archive,metadata)
    def test_live_changed_source_or_unknown_server_exit_is_rejected(self):
        for changes in ({'exit_code':1},{'exit_code':False},{'cleanup_finished':False},{'server_exit_code':None},{'server_exit_code':True},{'source_after_sha256':{}},{'runtime_dependencies_after':{}}):
            p,archive,metadata=proof_fixture(**changes)
            with self.subTest(changes=changes),self.assertRaises(AssertionError):admit_observation(json.dumps(p).encode(),archive,metadata)
    def test_incomplete_replay_or_changed_archive_cannot_enable_policy(self):
        for kind in ('status','budget','coverage','serial','checker','archive'):
            p,archive,metadata=proof_fixture()
            if kind=='status':p['status']='failed'
            elif kind=='budget':p['actual_workload_context_capacity']=4094
            elif kind=='coverage':p['profiles'][0]['raw_characterization']['actual_full4k_streams']=63
            elif kind=='serial':p['profiles'][0]['raw_characterization']['serial_after_comparisons']['text']['0']['completion_ids_exact']=False
            elif kind=='checker':p['replay_verifier_sha256']='b'*64
            else:archive+=b'changed'
            with self.subTest(kind=kind),self.assertRaises(AssertionError):admit_observation(json.dumps(p).encode(),archive,metadata)
    def test_identical_and_changed_text_both_record_exactly_without_invented_ids(self):
        result,control,body=stream_fixture();a=stream_comparison(result,control,'synthetic',body);self.assertTrue(a['serial_control_text_exact']);self.assertIsNone(a['first_different_character']);result['text']+=' 中文';b=stream_comparison(result,control,'synthetic',body);self.assertFalse(b['serial_control_text_exact']);self.assertEqual(b['first_different_character'],len(control['text']));self.assertNotEqual(b['stream_text_sha256'],b['serial_control_text_sha256']);self.assertFalse(b['streamed_ids_observed']);self.assertNotIn('stream_completion_token_ids',b)
    def test_lengths_stops_request_hash_and_typed_actual_control_ids_stay_strict(self):
        for kind in ('output','prompt','stop','hash','ignore','limit','thinking','temperature','ids'):
            result,control,body=stream_fixture()
            if kind=='output':result['generated_tokens']=126
            elif kind=='prompt':result['prompt_tokens']=3967
            elif kind=='stop':result['finish_reason']='stop'
            elif kind=='hash':result['request_sha256']='f'*64
            elif kind=='ignore':body['ignore_eos']=1
            elif kind=='limit':body['max_tokens']=True
            elif kind=='thinking':body['chat_template_kwargs']['enable_thinking']=True
            elif kind=='temperature':body['temperature']=False
            else:control['completion_token_ids'][27]=True
            with self.subTest(kind=kind),self.assertRaises(AssertionError):stream_comparison(result,control,'synthetic',body)
    def test_old_protocol_helpers_are_ast_identical_in_separate_benchmark(self):
        m=Path(__file__).resolve().parent
        for name in ('control_request','control_response','paired_control'):
            functions=[next(n for n in ast.parse((m/file).read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name==name) for file in ('benchmark_fixed_length_reference.py','benchmark_observed_output_reference.py')]
            self.assertEqual(ast.dump(functions[0],include_attributes=False),ast.dump(functions[1],include_attributes=False))


if __name__=='__main__':unittest.main()
