from pathlib import Path
import unittest
from .fp8_policy_cache_source import sources
from .o128_frontend_contract import BASE_FLAGS
class CacheSourceControls(unittest.TestCase):
    def test_all_sixty_numeric_sources_and_kernels_are_unchanged(self):
        parent,candidate,p=sources();self.assertEqual(len(parent),60);self.assertEqual(len(candidate),62);self.assertTrue(all(candidate[n]==b for n,b in parent.items()));self.assertEqual(set(candidate)-set(parent),{p['runtime_source'],p['new_header']})
    def test_policy_key_is_computed_before_cache_lookup_and_fails_closed(self):
        _,candidate,p=sources();text=candidate[p['runtime_source']].decode();self.assertLess(text.index('graphFingerprint = MakeGraphFingerprint('),text.index('LoadGraphCache(',text.index('graphFingerprint = MakeGraphFingerprint(')));self.assertIn('m_errorCode = "invalid_graph_policy"',text);self.assertEqual(text.count('material << Garnet::TRTGraphPolicyFingerprint();'),1)
    def test_registry_covers_current_profile_and_is_separate_from_quantizers(self):
        _,candidate,p=sources();header=candidate[p['new_header']].decode();self.assertTrue(all(header.count('"'+name+'"')==1 for name in BASE_FLAGS));self.assertIn('qwen38_graph_policy_v1',header);self.assertIn('std::strlen(value)',header);self.assertIn('enabled[4]&&!enabled[3]',header)
