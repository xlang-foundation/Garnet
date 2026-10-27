"""Source-only CPU controls; actual C++ behavior and model arithmetic are pending."""
from pathlib import Path
import unittest
from unittest.mock import patch
from .b4_prefill128_source import sources,M

class Source(unittest.TestCase):
    def test_frozen60_parent_only_seven_host_routes_and_one_header_change(self):
        read=Path.read_bytes
        def safe(p):
            if p.resolve().is_relative_to((M.parents[2]/'src').resolve()):raise AssertionError('Mutable installed production source read')
            return read(p)
        with patch.object(Path,'read_bytes',safe):old,new,p=sources()
        self.assertEqual((len(old),len(new),len(p['modified_files']),len(p['new_files'])),(60,61,7,1))
        for n,b in old.items():
            if n not in p['modified_files']:self.assertEqual(new[n],b)
    def test_CUDA_math_factories_wires_and_private_cache_are_identical(self):
        old,new,p=sources()
        for n,b in old.items():
            if n.endswith(('.cu','.cuh')) or 'plugin.' in n or 'canonical.' in n or n.endswith('cuda_compiled_graph.cpp'):self.assertEqual(new[n],b,n)
        self.assertEqual(p['new_context_factories'],0);self.assertEqual(p['new_CUDA_kernels'],0)
    def test_tampered_candidate_header_is_refused(self):
        read=Path.read_bytes
        def corrupt(p):
            b=read(p)
            return b+b' ' if p.name=='modelopt_fp8_batch4_prefill_profile.h' else b
        with patch.object(Path,'read_bytes',corrupt),self.assertRaises(AssertionError):sources()
if __name__=='__main__':unittest.main()
