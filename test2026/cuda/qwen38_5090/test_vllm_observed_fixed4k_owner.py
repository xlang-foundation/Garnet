"""CPU-only closure, complete dependencies and exact source-derivation controls."""
from pathlib import Path
import hashlib,json,tempfile,unittest,zipfile
from .vllm_observed_component_parent import JOB,closed_component
from .run_vllm_observed_fixed4k import job_path
from .prepare_vllm_observed_fixed4k_run import REQUIRED,require_observed_dependencies
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def fixture(root,**changes):
    folder=root/'results'/JOB;folder.mkdir(parents=True);w={'exit_code':0,'cleanup_finished':True,'source_sha256':{'s':'h'},'source_after_sha256':{'s':'h'},
        'runtime_dependencies':{'r':'h'},'runtime_dependencies_after':{'r':'h'},'environment':{'e':'h'},'environment_after':{'e':'h'},'sdk_before':{'x':'h'},'sdk_after':{'x':'h'}}
    w.update(changes);raw=json.dumps(w).encode();(folder/'workflow.json').write_bytes(raw);key='results/'+JOB+'/workflow.json';archive=root/'results'/(JOB+'-evidence.zip')
    with zipfile.ZipFile(archive,'x') as z:z.writestr(key,raw);z.writestr('evidence-manifest.json',json.dumps({'files':{key:sha(raw)}}))
    digest=sha(archive.read_bytes());q={'status':'prefill128_commit_component_full_state_replayed','archive_sha256':digest,'performance_admitted':False,'two_times_target_achieved':False,
        'full_state_replay':{'valid_cases':16,'invalid_cases':35,'graph_replays':3,'full_canonical_words_replayed':5448010944}}
    (root/'results'/(JOB+'-verified.json')).write_bytes(json.dumps(q).encode());return digest

class Controls(unittest.TestCase):
    def test_closed_component_and_source_runtime_sdk_faults(self):
        for changes,accepted in (({},True),({'exit_code':1},False),({'exit_code':False},False),({'cleanup_finished':False},False),
                ({'source_after_sha256':{}},False),({'runtime_dependencies_after':{}},False),({'environment_after':{}},False),({'sdk_after':{}},False)):
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);digest=fixture(root,**changes)
                if accepted:self.assertEqual(closed_component(root,digest)['exit_code'],0)
                else:
                    with self.assertRaises(AssertionError):closed_component(root,digest)
        with self.assertRaises(ValueError):closed_component(Path('.'),'f'*65)

    def test_fresh_ordinary_namespace(self):
        root=Path('synthetic');self.assertEqual(job_path(root,'vllm-observed-fixed4k-ordinary-attempt1'),root/'results/vllm-observed-fixed4k-ordinary-attempt1')
        for name in ('../x','vllm-fixed4k-ordinary-attempt1','vllm-observed-fixed4k-mtp-attempt1','sglang-observed-fixed4k-ordinary-attempt1','vllm-observed-fixed4k-ordinary-attempt0'):
            with self.assertRaises(ValueError):job_path(root,name)

    def test_every_runtime_method_is_pinned(self):
        files={'Garnet/test2026/cuda/qwen38_5090/'+n:'a'*64 for n in REQUIRED};require_observed_dependencies(files)
        for key in files:
            missing=dict(files);missing.pop(key)
            with self.assertRaises(ValueError):require_observed_dependencies(missing)
            drift=dict(files);drift[key]='b'*63
            with self.assertRaises(ValueError):require_observed_dependencies(drift)

    def test_exact_derivation_and_observed_raw_accounting(self):
        m=Path(__file__).resolve().parent
        for row in json.loads((m/'evidence/vllm-observed-fixed4k-pipeline-derivation.json').read_bytes())['derivations']:
            raw=(m/row['source']).read_bytes();self.assertEqual(sha(raw),row['source_sha256']);text=raw.decode()
            for step in row['replacements']:
                self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
            actual=(m/row['target']).read_bytes();self.assertEqual(actual,text.encode());self.assertEqual(sha(actual),row['target_sha256'])
        owner=(m/'run_vllm_observed_fixed4k.py').read_text();checker=(m/'verify_vllm_observed_fixed4k.py').read_text()
        self.assertIn('prepare_vllm_fixed4k_metadata.py',owner);self.assertIn('serve_vllm_fixed4k.sh',owner)
        self.assertIn('probe_vllm_observed_fixed4k.py',owner);self.assertNotIn('probe_reference_fixed4k.py',owner)
        self.assertIn('reference_vllm_observed_output',checker);self.assertIn('comparison_names=',checker)
        self.assertIn('installed_config_identity_sha256',checker);self.assertIn("counters['hit_delta']",checker)
        self.assertIn("('/reset_mm_cache',200)",checker);self.assertIn("('/reset_encoder_cache',200)",checker)

if __name__=='__main__':unittest.main()
