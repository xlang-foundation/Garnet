"""CPU construction/admission gates; no TensorRT build, CUDA or model execution."""
import copy,json,types,unittest
from pathlib import Path
from xModel.qwen35.vl_0_8b.bf16_kv_profile import validate_bf16_kv_decoder,validate_bf16_kv_program
from xModel.qwen35.vl_0_8b.joined_mlp_profile import validate_joined_mlp_profile
from xModel.qwen35.vl_0_8b.kv_bf16_study_layout import build_bf16_kv_study_layout
from xModel.qwen35.vl_0_8b.kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
from test2026.cuda.qwen38_5090.test_kv_delta_graph_contract import Trace,functions,member,MODEL,CONFIG

def options():return dict(external_compressed_weights=True,joined_mlp_projections=True)
def decoder_options():return dict(retain_linear_inputs=True,deferred_state_commit=True,shared_output_storage=True,host_embedding_lookup=True,**options())

def build_trace(batch,chunk,*,low_batch=False):
    trace=Trace();ns={'T':trace};source=(MODEL/'cuda_packed_program.py').read_bytes()
    functions(source,('packed_view','concat','flat_concat'),ns);functions((MODEL/'cuda_chunk.py').read_bytes(),('Qwen35CUDAChunk',),ns)
    builder=build_bf16_low_batch_layout if low_batch else build_bf16_kv_study_layout
    delta=builder(batch,chunk=chunk,retain_linear_inputs=chunk==4)
    p=types.SimpleNamespace(config=CONFIG,batch=batch,size=chunk,hidden_width=chunk*5120,state_width=delta['canonical_state_width'],
        layout=copy.deepcopy(delta['layers']),flat_concat=True,delta_layout=delta,bf16_kv_state=True,bf16_kv_low_batch_state=low_batch,
        external_compressed_weights=True,workspace=0,retain=chunk==4,register_delta_128=True,join_input_projections=False,
        joined_mlp_projections=True,kv_overlay_strategy=1,weight_names=[],weight_slots=[],weight_layers=[])
    inputs=[trace.input(n) for n in ('hidden','canonical','active','positions','slots')];weights=[]
    for layer in p.layout:
        prefix='model.language_model.layers.'+str(layer['index'])+'.';indices=[]
        suffixes=['mlp.gate_up_proj.weight','mlp.down_proj.weight']+(['linear_attn.'+n+'.weight' for n in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')] if layer['kind']=='linear_attention' else ['self_attn.'+n+'.weight' for n in ('q_proj','k_proj','v_proj','o_proj')])
        for suffix in suffixes:
            indices.append(len(p.weight_names));p.weight_names.append(prefix+suffix);slots=[]
            for role in (('packed','scales') if suffix.startswith('mlp.') else ('packed',)):
                slots.append(len(weights));weights.append(trace.input(prefix+suffix+'/'+role))
            p.weight_slots.append(slots)
        p.weight_layers.append(indices)
    forward=member(source,'CUDAPackedDecoderProgram','_forward',ns)(p);forward(*inputs,*weights)
    return trace,p

class BF16ModelContracts(unittest.TestCase):
    def test_four_program_profiles_and_two_decoder_profiles_admit(self):
        for batch in (4,8):
            validate_bf16_kv_decoder(True,batch,4096,32,4,CONFIG,**decoder_options())
            for chunk in (4,32):
                validate_bf16_kv_program(True,batch,4096,chunk,chunk==4,CONFIG,**options())
                validate_joined_mlp_profile(True,batch,4096,chunk,True,bf16_cache_profile=True)
    def test_legacy_joined_guards_are_not_relaxed(self):
        for batch,chunk in ((4,32),(8,4),(8,32)):
            with self.assertRaises(ValueError):validate_joined_mlp_profile(True,batch,4096,chunk,True)
    def test_flags_profiles_and_pinned_geometry_fail_before_gpu(self):
        for key in decoder_options():
            opts=decoder_options();opts[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):validate_bf16_kv_decoder(True,8,4096,32,4,CONFIG,**opts)
        for b,c,p,d in ((True,4096,32,4),(1,4096,32,4),(8,4095,32,4),(8,4096,128,4),(8,4096,32,8)):
            with self.subTest(shape=(b,c,p,d)),self.assertRaises(ValueError):validate_bf16_kv_decoder(True,b,c,p,d,CONFIG,**decoder_options())
        for key in ('hidden_size','num_attention_heads','linear_num_value_heads','head_dim','intermediate_size'):
            config=copy.deepcopy(CONFIG);config[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):validate_bf16_kv_program(True,8,4096,4,True,config,**options())
        for key,value in (('kv_delta_state',True),('embedded_projection_weights',True),('join_input_projections',True),('kv_overlay_strategy',0)):
            opts=options();opts[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_bf16_kv_program(True,8,4096,4,True,CONFIG,**opts)
    def test_bf16_full_attention_marker_offsets_and_no_fp32_kv_views(self):
        for batch in (4,8):
            for chunk in (4,32):
                trace,p=build_trace(batch,chunk)
                full=[r for r in trace.rows if r['name']=='qwen35_full_decoder_chunk'];self.assertEqual(len(full),16)
                for number,row in enumerate(full):
                    expected=p.delta_layout['layers'][number*4+3];kw=row['kw']
                    self.assertEqual(kw['canonical_bf16_kv_overlay'],1);self.assertNotIn('canonical_kv_overlay',kw)
                    self.assertEqual(kw['canonical_state_width'],106823680);self.assertEqual(kw['canonical_key_offset'],expected['offset'])
                    self.assertEqual(kw['canonical_value_offset'],expected['offset']+2097152);self.assertEqual(kw['kv_overlay_strategy'],1)
                views=[r for r in trace.rows if r['name']=='qwen35_packed_view']
                self.assertFalse(any(r['kw'].get('shape')==[batch,4,4096,128] or r['kw'].get('shape')==[batch,4,4096,256] for r in views))
                linear=[r for r in trace.rows if r['name']=='qwen35_linear_decoder_chunk'];self.assertEqual(len(linear),48)
                self.assertTrue(all(r['kw']['retain_linear_inputs']==int(chunk==4) for r in linear))
                self.assertTrue(all(r['kw']['register_delta_128']==1 for r in linear))
    def test_layout_metadata_not_mutated_by_symbolic_construction(self):
        for chunk in (4,32):
            _,p=build_trace(8,chunk)
            self.assertEqual(p.delta_layout,build_bf16_kv_study_layout(8,chunk=chunk,retain_linear_inputs=chunk==4))
    def test_close_pending_transaction_leaves_bf16_helper_open(self):
        from test2026.cuda.qwen38_5090.test_kv_delta_transactions import decoder
        d=decoder(deferred=True);events=d.events
        d.bf16_prefill_commit=types.SimpleNamespace(close=lambda:events.append('closed-bf16-helper'))
        with self.assertRaises(ValueError):d.close()
        self.assertNotIn('closed-bf16-helper',events)

if __name__=='__main__':unittest.main()
