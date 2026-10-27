import unittest
from pathlib import Path
from unittest.mock import patch
from .o128_canonical_source import sources,M
class O128SourceTests(unittest.TestCase):
    def test_exact60_source_uses_frozen57_parent_and_preserves_legacy_math(self):
        read=Path.read_bytes
        def safe(p):
            if p.resolve().is_relative_to((M.parents[2]/'src').resolve()):raise AssertionError('Mutable installed native source read')
            return read(p)
        with patch.object(Path,'read_bytes',safe):old,new,p=sources()
        self.assertEqual((len(old),len(new),len(p['new_files']),len(p['modified_files'])),(57,60,3,5))
        for n,b in old.items():
            if n not in p['modified_files']:self.assertEqual(new[n],b)
        self.assertEqual(p['new_context_factories'],0);self.assertEqual(p['new_CUDA_kernels'],0)
    def test_shared_M128_SDK_factory_and_8MiB_workspace_contract_unchanged(self):
        old,new,p=sources();prefix='src/backends/tensorrt/'
        for n in ('modelopt_fp8_m128_canonical.cpp','modelopt_fp8_m128_canonical.h','modelopt_fp8_m128_plugin.cpp','modelopt_fp8_m128_plugin.h','modelopt_fp8_m128_profile.h'):self.assertEqual(new[prefix+n],old[prefix+n])
        h=new[prefix+'modelopt_fp8_o128_plugin.h'];cpp=new[prefix+'modelopt_fp8_o128_plugin.cpp'];self.assertIn(b'#include "modelopt_fp8_m128_canonical.h"',h);self.assertIn(b'std::shared_ptr<ModelOptFP8M128Context>',h);self.assertIn(b'CreateModelOptFP8M128Context()',cpp);self.assertIn(b'RunModelOptFP8M128Matmul(context_,input[1],packed,output[0],128,5120,6144',cpp);self.assertIn(b'ModelOptFP8M128BlasWorkspace',cpp);self.assertIn(b'ModelOptFP8M128BlasWorkspace=8ULL<<20',new[prefix+'modelopt_fp8_m128_canonical.h']);self.assertNotIn(b'CreateModelOptFP8O128Context',cpp)
    def test_separate_O_wire_counters_and_original_quantizer_scaling(self):
        old,new,p=sources();prefix='src/backends/tensorrt/';h=new[prefix+'modelopt_fp8_o128_plugin.h'];cpp=new[prefix+'modelopt_fp8_o128_plugin.cpp'];self.assertIn(b'GarnetModelOptFP8O128Canonical',h);self.assertIn(b'0x4f313238u',h);self.assertIn(b'QuantizeModelOptFP8Activation(input[0],packed,state_.inputScale,128,6144,stream)',cpp);self.assertIn(b'state_.weightScale*state_.inputScale',cpp)
        self.assertIn(b'GARNET_FP8_FULL_ATTN_O128_CANONICAL',new[prefix+'modelopt_fp8_lowering.cpp']);self.assertIn(b'native_fp8_o128_canonical_projections',new[prefix+'cuda_compiled_graph.cpp'])
        for n in p['modified_files']:
            raw=new[n]
            for c in reversed(p['compositions'][n]):self.assertEqual(raw.count(c['new'].encode()),1);raw=raw.replace(c['new'].encode(),c['old'].encode())
            self.assertEqual(raw,old[n])
    def test_text_O_only_strict_geometry_private_cache_unchanged(self):
        old,new,p=sources();prefix='src/backends/tensorrt/';raw=new[prefix+'modelopt_fp8_o128_profile.h']
        for value in (b'layer=3;layer<64;layer+=4',b'name==prefix+"o_proj.weight"',b'batch!=1||tokens!=128||inputRows!=128',b'outputRows!=5120||columns!=6144',b'GARNET_FP8_FULL_ATTN_O128_CANONICAL must be0 or1'):self.assertIn(value,raw)
        for value in (b'q_proj.weight',b'k_proj.weight',b'v_proj.weight'):self.assertNotIn(value,raw)
        before=old[prefix+'cuda_compiled_graph.cpp'];after=new[prefix+'cuda_compiled_graph.cpp'];start=before.index(b'CUDACompiledGraph::CUDACompiledGraph');end=before.index(b'CUDACompiledGraph::~CUDACompiledGraph');self.assertIn(before[start:end],after)
