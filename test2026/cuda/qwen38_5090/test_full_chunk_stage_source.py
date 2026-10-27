"""Exact observation-only composition and unchanged native guards/math."""
import unittest
from .full_chunk_stage_source import sources
class FullChunkStageSourceTests(unittest.TestCase):
    def test_all51_sources_exact_composition_and_every_old_guard_preserved(self):
        old,new,p=sources();self.assertEqual(len(new),51)
        for n,b in old.items():
            if n not in p['modified_files']:self.assertEqual(new[n],b)
        # Undo only declared additive observation replacements to obtain exact bytes.
        for n,changes in p['derivations'].items():
            b=new[n]
            for c in reversed(changes):self.assertEqual(b.count(c['new'].encode()),1);b=b.replace(c['new'].encode(),c['old'].encode())
            self.assertEqual(b,old[n])
        cpp=new['src/backends/tensorrt/trt_builder.cpp'].decode();prior=old['src/backends/tensorrt/trt_builder.cpp'].decode()
        start=prior.index('                    auto* stageAttribute=FindKeyword(kwParams,"diagnostic_full_stages")');end=prior.index('                    std::vector<ITensor*> diagnosticStages;',start)
        self.assertIn(prior[start:end],cpp)
        lower=new['src/backends/tensorrt/qwen35_lowering.h'].decode()
        self.assertIn('||diagnosticStages))',lower);self.assertIn('auto* capturedStages=diagnosticStages?diagnosticStages:chunkStages;',lower)
        # Existing INT4-only projection guard and unrelated native kernels/plugins are exact.
        for n in old:
            if n.endswith(('.cu','.cuh')) or 'plugin' in n:self.assertEqual(old[n],new[n])

    def test_new_observation_requires_explicit_typed_policy_and_separate_counter(self):
        _,new,_=sources();cpp=new['src/backends/tensorrt/trt_builder.cpp'].decode()
        for text in ('witness_full_chunk_stages','GARNET_FULL_CHUNK_STAGE_WITNESS','IsInt64()','IsUInt64()','ToUInt64()>1','RequireFullChunkStageGeometry','RequireFullChunkStageWitness','chunkWitness?&chunkStageOutputs:nullptr','chunkStageOutputs.size()!=10','width>2097152/shape[i]','++nativeFullChunkStageWitnessCount_'):
            self.assertIn(text,cpp)
        graph=new['src/backends/tensorrt/cuda_compiled_graph.cpp'].decode();self.assertIn('info["native_full_chunk_stage_witnesses"]=nativeFullChunkStageWitnessCount_;',graph)
        self.assertIn('nativeFullChunkStageWitnessCount_=0;',graph)
