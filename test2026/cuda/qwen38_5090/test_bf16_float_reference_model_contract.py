"""CPU admission and symbolic full-cache contracts; no CUDA/TRT execution."""
import copy,types,unittest
from xModel.qwen35.vl_0_8b.bf16_float_reference_profile import validate_bf16_float_reference_program as program_gate,validate_bf16_float_reference_decoder as decoder_gate
from xModel.qwen35.vl_0_8b.bf16_kv_profile import validate_bf16_kv_decoder
from xModel.qwen35.vl_0_8b.joined_mlp_profile import validate_joined_mlp_profile
from test2026.cuda.qwen38_5090.test_bf16_kv_model_contract import options,decoder_options
from test2026.cuda.qwen38_5090.test_kv_delta_graph_contract import Trace,functions,member,MODEL,CONFIG,target_trace


def reference_trace(chunk):
    trace=Trace();ns={'T':trace};source=(MODEL/'cuda_packed_program.py').read_bytes()
    functions(source,('decoder_state_layout','packed_view','concat','flat_concat'),ns)
    functions((MODEL/'cuda_chunk.py').read_bytes(),('Qwen35CUDAChunk',),ns)
    layout,width=ns['decoder_state_layout'](CONFIG,4,4096)
    p=types.SimpleNamespace(config=CONFIG,batch=4,size=chunk,hidden_width=chunk*5120,state_width=width,
        layout=layout,flat_concat=True,delta_layout=None,bf16_kv_state=False,bf16_kv_float_reference=True,
        external_compressed_weights=True,workspace=0,retain=chunk==4,register_delta_128=True,join_input_projections=False,
        joined_mlp_projections=True,kv_overlay_strategy=1,weight_names=[],weight_slots=[],weight_layers=[])
    args=[trace.input(n) for n in ('hidden','state','active','positions','slots')];weights=[]
    for layer in layout:
        layer['payload_width']=0;indices=[];prefix='model.language_model.layers.'+str(layer['index'])+'.'
        suffixes=['mlp.gate_up_proj.weight','mlp.down_proj.weight']+(['linear_attn.'+n+'.weight' for n in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj')] if layer['kind']=='linear_attention' else ['self_attn.'+n+'.weight' for n in ('q_proj','k_proj','v_proj','o_proj')])
        for suffix in suffixes:
            indices.append(len(p.weight_names));p.weight_names.append(prefix+suffix);slots=[]
            for role in (('packed','scales') if suffix.startswith('mlp.') else ('packed',)):
                slots.append(len(weights));weights.append(trace.input(prefix+suffix+'/'+role))
            p.weight_slots.append(slots)
        p.weight_layers.append(indices)
    member(source,'CUDAPackedDecoderProgram','_forward',ns)(p)(*args,*weights)
    return trace,p,ns


class BF16FloatReferenceModelContracts(unittest.TestCase):
    def test_two_reference_programs_and_decoder_admit(self):
        decoder_gate(True,4,4096,32,4,CONFIG,**decoder_options())
        for chunk in (4,32):
            program_gate(True,4,4096,chunk,chunk==4,CONFIG,**options())
            validate_joined_mlp_profile(True,4,4096,chunk,True,bf16_float_reference=True)

    def test_reference_rejects_unqualified_geometry_and_flags(self):
        for batch,cap,chunk in ((8,4096,4),(1,4096,4),(True,4096,4),(4,4095,4),(4,4096,128),(4,4096,4.0)):
            with self.subTest(shape=(batch,cap,chunk)),self.assertRaises(ValueError):program_gate(True,batch,cap,chunk,chunk==4,CONFIG,**options())
        for key,value in [('bf16_kv_state',True),('bf16_kv_state',0),('kv_delta_state',True),('embedded_projection_weights',True),('join_input_projections',True),('kv_overlay_strategy',0),('external_compressed_weights',False),('joined_mlp_projections',False)]:
            opts=options();opts[key]=value
            with self.subTest(flag=key),self.assertRaises(ValueError):program_gate(True,4,4096,4,True,CONFIG,**opts)
        for enabled in (0,1,None,'true'):
            with self.assertRaises(ValueError):program_gate(enabled,4,4096,4,True,CONFIG,**options())
        for chunk in (4,32):
            with self.assertRaises(ValueError):program_gate(True,4,4096,chunk,chunk!=4,CONFIG,**options())

    def test_pinned_geometry_rejections(self):
        for key in ('hidden_size','num_attention_heads','num_key_value_heads','head_dim','intermediate_size','linear_num_value_heads'):
            config=copy.deepcopy(CONFIG);config[key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):program_gate(True,4,4096,4,True,config,**options())
        config=copy.deepcopy(CONFIG);config['layer_types'][3]='linear_attention'
        with self.assertRaises(ValueError):program_gate(True,4,4096,4,True,config,**options())

    def test_decoder_admission_precedes_gpu_construction(self):
        def forbidden():raise AssertionError('GPU IO reached during rejected construction')
        ns={'validate_bf16_float_reference_decoder':decoder_gate,'validate_bf16_kv_decoder':validate_bf16_kv_decoder,'garnet':types.SimpleNamespace(cuda_tensor_io=forbidden)}
        init=member((MODEL/'cuda_packed_decoder.py').read_bytes(),'CUDAPackedDecoder','__init__',ns)
        good=dict(batch=4,capacity=4096,prefill_chunk=32,decode_chunk=4,bf16_kv_float_reference=True,**decoder_options())
        for key,value in [('batch',8),('bf16_kv_state',True),('kv_delta_state',True),('prefill_chunk',4),('bf16_kv_float_reference',1),('retain_linear_inputs',False),('deferred_state_commit',False),('shared_output_storage',False),('host_embedding_lookup',False)]:
            opts=dict(good);opts[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):init(types.SimpleNamespace(),'unused',CONFIG,**opts)

    def test_reference_profile_does_not_relax_legacy_joined_guards(self):
        for batch,chunk in ((4,32),(8,4),(8,32)):
            with self.assertRaises(ValueError):validate_joined_mlp_profile(True,batch,4096,chunk,True)
        for batch in (1,8):
            with self.assertRaises(ValueError):validate_joined_mlp_profile(True,batch,4096,4,True,bf16_float_reference=True)
        with self.assertRaises(ValueError):validate_joined_mlp_profile(True,4,4096,4,True,bf16_float_reference=True,bf16_cache_profile=True)

    def test_symbolic_reference_uses_full_float_caches_without_compact_bindings(self):
        for chunk in (4,32):
            trace,p,_=reference_trace(chunk)
            self.assertEqual(p.state_width,173932544);self.assertIsNone(p.delta_layout)
            full=[r for r in trace.rows if r['name']=='qwen35_full_decoder_chunk'];self.assertEqual(len(full),16)
            for row in full:
                self.assertEqual(row['kw']['bf16_kv_float_reference'],1)
                self.assertNotIn('canonical_bf16_kv_overlay',row['kw']);self.assertNotIn('canonical_kv_overlay',row['kw'])
                self.assertNotIn('canonical_key_offset',row['kw'])
            self.assertEqual(sum(r['name']=='qwen35_attention_bind_key' for r in trace.rows),16)
            self.assertEqual(sum(r['name']=='qwen35_attention_bind_value' for r in trace.rows),16)
            self.assertFalse(any(r['name']=='qwen35_attention_bind_canonical' for r in trace.rows))
            cache_views=[r for r in trace.rows if r['name']=='qwen35_packed_view' and r['args'][0].index==1 and r['kw']['shape']==[4,4,4096,256]]
            self.assertEqual(len(cache_views),32)
            expected=[offset for l in p.layout if l['kind']=='full_attention' for offset in (l['offset'],l['offset']+l['state1_width'])]
            self.assertEqual([r['kw']['start'] for r in cache_views],expected)
            linear=[r for r in trace.rows if r['name']=='qwen35_linear_decoder_chunk'];self.assertEqual(len(linear),48)
            self.assertTrue(all('bf16_kv_float_reference' not in r['kw'] and r['kw']['register_delta_128']==1 and r['kw']['retain_linear_inputs']==int(chunk==4) for r in linear))

    def test_disabled_graph_retains_original_digest(self):
        import json
        from pathlib import Path
        expected=json.loads((Path(__file__).parent/'evidence/kv-delta-disabled-graph-golden.json').read_bytes())
        trace,_,_=target_trace();self.assertEqual(trace.digest(),expected['trace_sha256']);self.assertEqual(len(trace.rows),expected['trace_nodes'])

    def test_direct_chunk_rejects_compact_or_linear_reference_bindings(self):
        trace,p,ns=reference_trace(4);symbols=[trace.input(n) for n in ('x','key','value','active','positions','slots')]
        for extras,kind in [({'canonical_kv_state':symbols[1],'canonical_kv_layout':{}},'full_attention'),({},'linear_attention'),({'bf16_kv_float_reference':1},'full_attention')]:
            kw={'bf16_kv_float_reference':True};kw.update(extras)
            with self.subTest(kind=kind,extras=extras),self.assertRaises(ValueError):ns['Qwen35CUDAChunk'](*symbols,CONFIG,3,kind,**kw)

if __name__=='__main__':unittest.main()
