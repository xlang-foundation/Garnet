import unittest
from pathlib import Path
from unittest.mock import patch
from .q128_canonical_source import sources,M
class Q128SourceTests(unittest.TestCase):
    def test_exact57source_derivation_never_reads_mutable_installed_headers(self):
        read=Path.read_bytes
        def safe(p):
            if p.resolve().is_relative_to((M.parents[2]/'src').resolve()):raise AssertionError('Mutable installed native source read')
            return read(p)
        with patch.object(Path,'read_bytes',safe):old,new,p=sources()
        self.assertEqual((len(old),len(new),len(p['modified_files']),len(p['new_files'])),(51,57,5,6))
        for n,b in old.items():
            if n.endswith(('.cu','.cuh')) or 'plugin' in n or 'qwen35_lowering' in n:self.assertEqual(new[n],b)
    def test_live_SDK_object_and_only_observed_seven_supported_attributes(self):
        _,new,p=sources();raw=new['src/backends/tensorrt/modelopt_fp8_q128_canonical.cpp']
        self.assertEqual(raw.count(b'QConfigEqual<'),7);self.assertEqual(raw.count(b'trace.api("cublasLtMatmulAlgoCheck"'),3)
        for text in (b'INNER_SHAPE_ID',b'CLUSTER_SHAPE_ID',b'AlgoConfigSetAttribute',b'AlgoInit',b'expectedOpaque',b'std::memcmp'):self.assertNotIn(text,raw)
        self.assertIn(b'context->info.opaque=choices[i].algo',raw)
        self.assertIn(b'canonical.input,CUDA_R_8F_E4M3,5120,16,5120',raw);self.assertIn(b'canonical.controlInput,CUDA_R_8F_E4M3,5120,32,5120',raw)
        self.assertEqual(p['observed_supported_public_attributes'],dict(ID=67,TILE_ID=13,STAGES_ID=17,CTA_SWIZZLING=1,CUSTOM_OPTION=18,SPLITK_NUM=1,REDUCTION_SCHEME=0))
    def test_separate_wire_counter_defaultoff_route_and_original_quantizer(self):
        old,new,p=sources();prefix='src/backends/tensorrt/';h=new[prefix+'modelopt_fp8_q128_plugin.h'];cpp=new[prefix+'modelopt_fp8_q128_plugin.cpp']
        self.assertIn(b'GarnetModelOptFP8Q128Canonical',h);self.assertIn(b'0x51313238u',h);self.assertIn(b'QuantizeModelOptFP8Activation(input[0],packed,state_.inputScale,128,5120,stream)',cpp);self.assertIn(b'state_.weightScale*state_.inputScale',cpp)
        self.assertIn(b'GARNET_FP8_FULL_ATTN_Q128_CANONICAL',new[prefix+'modelopt_fp8_lowering.cpp']);self.assertIn(b'native_fp8_q128_canonical_projections',new[prefix+'cuda_compiled_graph.cpp'])
        for n in p['modified_files']:
            raw=new[n]
            for c in reversed(p['compositions'][n]):self.assertEqual(raw.count(c['new'].encode()),1);raw=raw.replace(c['new'].encode(),c['old'].encode())
            self.assertEqual(raw,old[n])
    def test_text_Q_only_and_private_cache_isolation_preserved(self):
        old,new,p=sources();raw=new['src/backends/tensorrt/modelopt_fp8_q128_profile.h']
        for text in (b'layer=3;layer<64;layer+=4',b'name==prefix+"q_proj.weight"',b'batch!=1||tokens!=128||inputRows!=128',b'outputRows!=12288||columns!=5120'):self.assertIn(text,raw)
        for text in (b'k_proj.weight',b'v_proj.weight'):self.assertNotIn(text,raw)
        before=old['src/backends/tensorrt/cuda_compiled_graph.cpp'];after=new['src/backends/tensorrt/cuda_compiled_graph.cpp'];start=before.index(b'CUDACompiledGraph::CUDACompiledGraph');end=before.index(b'CUDACompiledGraph::~CUDACompiledGraph');self.assertIn(before[start:end],after)
