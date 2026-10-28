"""Synthetic CPU fault controls; these arrays are never hardware measurements."""
import copy,hashlib,struct,unittest
import numpy as np
from .fa3_projection_witness import STAGES,replay,check_graph
sha=lambda b:hashlib.sha256(b).hexdigest()
def fixture():
    files={};counter=0
    def store(raw):
        nonlocal counter
        name='synthetic-'+str(counter)+'.f32';counter+=1;files[name]=raw
        return dict(words=len(raw)//4,sha256=sha(raw),parts=[dict(file=name,offset=0,words=len(raw)//4,sha256=sha(raw))])
    common=np.repeat(np.arange(1,129,dtype='<f4'),5120).tobytes();report=dict(status='actual_FA3_projection_witness_collected',model_or_performance_qualified=False,witness_policy='1',source_input_metadata={'sha256':sha(common)},input_fixture_sha256=sha(common),common_input=store(common),cases=[],weight_metadata={})
    for kind in ('k','v'):report['weight_metadata']['model.language_model.layers.3.self_attn.'+kind+'_proj.weight']=dict(logical_shape=[1024,5120],packed_shape=[1024,1280],packed_dtype='INT32',packed_bytes=5242880,recipe='modelopt_fp8_w8a8',scale_bytes=0,scale_shape=[],joined_mlp=False,weight_scale=.5,input_scale=.25)
    for t in (32,128):
        stats=dict(checkpoint_weight_precision='modelopt_mixed',native_fp8_projection_weights=2,native_nvfp4_projection_weights=0,native_int4_g128_projection_weights=0,external_compressed_weight_inputs=0,external_bf16_weight_inputs=0,native_fp8_m128_canonical_projections=0,native_fp8_z128_canonical_projections=0,native_fp8_z128_reconstructed_projections=0,device_output_bytes=t*7168*4,graph_capture_failures=0,graph_captures=1,graph_replays=2)
        row=dict(tokens=t,graph=stats,repeats=[]);report['cases'].append(row)
        for repeat in (0,1):
            chunks=[];row['repeats'].append(chunks)
            for start in range(0,128,t):
                hidden=common[start*5120*4:(start+t)*5120*4];norm=hidden;key=np.repeat(np.arange(start+1,start+t+1,dtype='<f4'),1024).tobytes();value=(np.frombuffer(key,dtype='<f4')*2).tobytes()
                chunks.append(dict(start=start,repeat=repeat,input_hidden=store(hidden),input_hidden_after=store(hidden),arrays=dict(normalized=store(norm),key=store(key),value=store(value))))
    return report,files,common,store
class Controls(unittest.TestCase):
    def test_all_complete_arrays_replay_and_exact_input_scope(self):
        r,f,c,_=fixture();q=replay(r,f.__getitem__,c);self.assertEqual(q['finite_array_entries'],51);self.assertEqual(q['complete_output_F32_words_replayed'],3670016);self.assertTrue(q['normalized_exact'] and q['key_value_both_exact']);self.assertFalse(q['model_or_quality_or_performance_qualified'])
    def test_rehashed_projection_and_norm_differences_are_retained_without_false_attribution(self):
        for stage in ('key','value','normalized'):
            r,f,c,store=fixture();entry=r['cases'][1]['repeats'][0][0]['arrays'][stage];raw=bytearray(f[entry['parts'][0]['file']]);raw[-4]^=1;new=store(bytes(raw))
            for sequence in r['cases'][1]['repeats']:sequence[0]['arrays'][stage]=new
            q=replay(r,f.__getitem__,c);self.assertEqual(q['stages'][stage]['changed_F32_words'],1);self.assertFalse(q['stages'][stage]['byte_exact']);self.assertEqual(q['actual_projection_difference_with_identical_normalized_input'],stage!='normalized');self.assertFalse(q['full_attention_cache_cause_established'])
    def test_input_mutation_repeat_drift_and_nonfinite_are_rejected(self):
        for fault in ('input','repeat','nonfinite','truncated'):
            r,f,c,store=fixture();chunk=r['cases'][1]['repeats'][0][0]
            if fault=='input':chunk['input_hidden_after']=store(bytes(len(c)))
            elif fault=='repeat':chunk['arrays']['key']=store(struct.pack('<f',2)*131072)
            elif fault=='nonfinite':chunk['arrays']['key']=store(struct.pack('<f',float('inf'))*131072)
            else:chunk['arrays']['key']=store(bytes(4))
            with self.assertRaises(AssertionError):replay(r,f.__getitem__,c)
    def test_counter_weight_scale_and_profile_mutations_are_rejected(self):
        r,f,c,_=fixture()
        for key,value in [('native_fp8_projection_weights',1),('native_nvfp4_projection_weights',1),('native_fp8_z128_reconstructed_projections',True),('device_output_bytes',0),('graph_captures',0)]:
            bad=copy.deepcopy(r);bad['cases'][1]['graph'][key]=value
            with self.assertRaises(AssertionError):replay(bad,f.__getitem__,c)
        for key,value in [('input_scale',float('nan')),('weight_scale',False),('logical_shape',[5120,1024])]:
            bad=copy.deepcopy(r);next(iter(bad['weight_metadata'].values()))[key]=value
            with self.assertRaises(AssertionError):replay(bad,f.__getitem__,c)
if __name__=='__main__':unittest.main()
