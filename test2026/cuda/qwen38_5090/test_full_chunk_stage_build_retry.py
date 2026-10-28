from pathlib import Path
import hashlib,json,unittest
from unittest.mock import patch
from .full_chunk_stage_source import sources
M=Path(__file__).resolve().parent;R=M.parents[2]
class FullChunkStageBuildRetryTests(unittest.TestCase):
    def test_missing_or_modified_installed_headers_do_not_change_frozen_source_and_parent_installs_before_old_controls(self):
        read=Path.read_bytes
        def frozen(path):
            if path.is_relative_to(R/'src') or path.is_relative_to(R/'xModel'):raise AssertionError('Mutable installed production source was read')
            return read(path)
        with patch.object(Path,'read_bytes',frozen):old,new,p=sources()
        self.assertEqual(len(old),50);self.assertEqual(len(new),51)
        record=json.loads((M/'evidence/full-chunk-stage-build-retry-derivation.json').read_bytes())
        for row in record['derivations']:
            raw=(M/row['source']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),row['source_sha256'])
            for step in row['steps']:self.assertEqual(raw.count(step['old'].encode()),step['count']);raw=raw.replace(step['old'].encode(),step['new'].encode())
            self.assertEqual(raw,(M/row['target']).read_bytes());self.assertEqual(hashlib.sha256(raw).hexdigest(),row['target_sha256'])
        owner=(M/'run_full_chunk_stage_build_attempt2.py').read_text()
        self.assertLess(owner.index('for n,b in parent.items():installed[n]=b;'),owner.index("],'cpu-before',"))
        self.assertLess(owner.index("],'cpu-before',"),owner.index('for n,b in candidate.items():installed[n]=b;'))
        self.assertIn("w['parent_installed_source']={n:sha(ROOT/n) for n in inventory}",owner)
        self.assertTrue(record['all51native_candidate_bytes_unchanged'])
