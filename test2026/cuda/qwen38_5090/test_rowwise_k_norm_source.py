from pathlib import Path
import json,unittest
import numpy as np
from .rowwise_k_norm_source import sources
M=Path(__file__).resolve().parent

class SourceControls(unittest.TestCase):
    def test_exact87_and_only_two_host_sources_plus_one_guard(self):
        parent,candidate,p=sources();self.assertEqual((len(parent),len(candidate)),(86,87))
        self.assertEqual(set(p['modified_files']),{'src/backends/tensorrt/qwen35_lowering.h','src/backends/tensorrt/trt_builder.cpp'})
        self.assertEqual(p['new_files'],['src/backends/tensorrt/qwen38_rowwise_k_norm_profile.h'])
        for name,raw in parent.items():
            if name not in p['modified_files']:self.assertEqual(candidate[name],raw)

    def test_exact_reverse_derivations_preserve_every_old_guard_and_math_branch(self):
        parent,candidate,p=sources()
        for row in p['derivations']:
            raw=candidate[row['target']]
            for c in reversed(row['changes']):
                self.assertEqual(raw.count(c['new'].encode()),c['count']);raw=raw.replace(c['new'].encode(),c['old'].encode())
            self.assertEqual(raw,parent[row['source']])

    def test_explicit_off_default_and_no_observer_or_dtype_conversion_in_new_norm_branch(self):
        _,candidate,_=sources();lower=candidate['src/backends/tensorrt/qwen35_lowering.h'].decode();builder=candidate['src/backends/tensorrt/trt_builder.cpp'].decode()
        self.assertIn('bool rowwiseKDecodeNorm=false',lower);self.assertIn('rowwiseKAttribute?rowwiseKAttribute->second.ToLongLong():0',builder)
        self.assertIn('GARNET_ROWWISE_K_DECODE_NORM',builder);self.assertIn('rowwise_full_decode_k_norm',builder)
        start=lower.index('            T* k=nullptr;');end=lower.index('            if(decode35Stages)',start);branch=lower[start:end]
        self.assertIn('rms(slice(shaped,0,row,1),weight(prefix+".k_norm.weight"),eps,true)',branch)
        self.assertIn('else k=rms(reshape(projectedK,{batch,tokens,kv,dim}),weight(prefix+".k_norm.weight"),eps,true)',branch)
        self.assertNotIn('cast(',branch);self.assertNotIn('decode35Stages',branch);self.assertIn('k=cat(normalizedRows,0)',branch)

    def test_pinned_model_geometry_independently_derives_all16_cache_offsets(self):
        cfg=json.loads((M/'evidence/nvfp4-config.json').read_bytes())['text_config']
        self.assertEqual((cfg['num_hidden_layers'],cfg['hidden_size'],cfg['head_dim'],cfg['num_key_value_heads']),(64,5120,256,4))
        conv=(2*cfg['linear_num_key_heads']*cfg['linear_key_head_dim']+cfg['linear_num_value_heads']*cfg['linear_value_head_dim'])*cfg['linear_conv_kernel_dim']
        recurrent=cfg['linear_num_value_heads']*cfg['linear_key_head_dim']*cfg['linear_value_head_dim'];linear=conv+recurrent
        key=cfg['num_key_value_heads']*4096*cfg['head_dim']//2;stride=3*linear+2*key
        self.assertEqual((3*linear,key,stride,16*stride),(2482176,2097152,6676480,106823680))
        offsets=[(4*i+3,i*stride+3*linear,i*stride+3*linear+key) for i in range(16)]
        self.assertEqual(offsets[8],(35,55894016,57991168));self.assertEqual(offsets[-1][-1]+key,106823680)

    def test_same_high_precision_RMS_equation_for_full_batch_and_each_row(self):
        x=np.random.default_rng(5090).normal(size=(4,4,4,256));scale=np.linspace(-.5,.5,256,dtype=np.float64)
        def rms(a):return (a/np.sqrt(np.mean(a*a,axis=-1,keepdims=True)+1e-6))*(1+scale)
        expected=rms(x);split=np.concatenate([rms(x[row:row+1]) for row in range(4)],axis=0)
        np.testing.assert_array_equal(expected,split)
        # Establishes equation/indexing only; TensorRT rounding and model repair
        # require actual native controls and full-context model runs.

if __name__=='__main__':unittest.main()
