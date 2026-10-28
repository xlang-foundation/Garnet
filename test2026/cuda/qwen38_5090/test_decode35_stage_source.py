"""Source/ABI controls; actual native header contracts are tested on remote CPU."""
from pathlib import Path
import unittest
from .decode35_stage_source import sources

class Contract(unittest.TestCase):
    def test_exact_composition_all_original_operators_guards_and_CUDA_retained(self):
        old,new,p=sources()
        self.assertEqual(len(new),86)
        for n in old:
            if n not in p['modified_files']:self.assertEqual(old[n],new[n])
        for row in p['derivations']:
            raw=new[row['target']]
            for c in reversed(row['changes']):
                self.assertEqual(raw.count(c['new'].encode()),c['count'])
                raw=raw.replace(c['new'].encode(),c['old'].encode())
            self.assertEqual(raw,old[row['source']])

    def test_new_metadata_and_policy_are_separate_default_off(self):
        _,new,_=sources();s=new['src/backends/tensorrt/trt_builder.cpp'].decode()
        self.assertIn('decode35Attribute?decode35Attribute->second.ToLongLong():0',s)
        self.assertIn('decode35Attribute->second.IsInt64()||decode35Attribute->second.IsUInt64()',s)
        self.assertIn('decode35Attribute->second.ToUInt64()>1',s)
        self.assertIn('decode35Witness!=0&&decode35Witness!=1',s)
        self.assertIn('std::getenv("GARNET_DECODE35_STAGE_WITNESS")',s)
        self.assertIn('decode35Witness?&decode35StageOutputs:nullptr',s)
        self.assertIn('chunkWitness?&chunkStageOutputs:nullptr,decode35Witness?',s)

    def test_exact_internal_operands_no_rerun_or_precision_change(self):
        _,new,p=sources();s=new['src/backends/tensorrt/qwen35_lowering.h'].decode()
        self.assertIn('std::vector<T*>* decode35Stages=nullptr',s)
        self.assertEqual(s.count('if(decode35Stages)*decode35Stages={x,normalized,qg,projectedK,projectedV};'),1)
        self.assertEqual(s.count('if(decode35Stages)decode35Stages->push_back(k);'),1)
        self.assertEqual(s.count('if(decode35Stages)decode35Stages->push_back(key);'),1)
        self.assertEqual(p['stage_names'],['masked_input','input_normalized','projected_Q_gate','projected_K','projected_V','K_normalized_pre_RoPE','K_post_RoPE'])
        self.assertNotIn('nativeDecode35',new['src/backends/tensorrt/trt_builder.cpp'].decode())

    def test_fixed_float_stage_sizes_checked_before_trace(self):
        _,new,p=sources();s=new['src/backends/tensorrt/trt_builder.cpp'].decode()
        self.assertEqual(p['stage_widths'],[20480,20480,49152,4096,4096,4096,4096])
        self.assertIn('value->getType()!=DataType::kFLOAT',s)
        self.assertIn('width>Decode35StageWidths[stage]/shape[i]',s)
        self.assertIn('width!=Decode35StageWidths[stage]',s)
        self.assertIn('lastOutput=ops.cat(parts,1);TraceDecode35StageWitness(dimensions[0]);',s)

    def test_corrupt_candidate_rejected(self):
        _,new,p=sources()
        for target in p['modified_files']+p['new_files']:
            def corrupt(n):return new[n]+(b'\n// corrupt' if n==target else b'')
            with self.assertRaises(AssertionError):sources(corrupt)

if __name__=='__main__':unittest.main()
