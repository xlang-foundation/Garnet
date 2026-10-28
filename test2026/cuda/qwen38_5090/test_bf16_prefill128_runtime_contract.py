"""Pure CPU T128 routing/flag/layout checks; no native runtime or GPU execution."""
from pathlib import Path
import ast,copy,hashlib,json,sys,types,unittest
from unittest.mock import patch
from .test_kv_delta_graph_contract import Trace,functions,member,MODEL,CONFIG,ROOT
from .bf16_prefill128_layout import build_bf16_prefill128_layout as independent_layout
from xModel.qwen35.vl_0_8b.kv_bf16_prefill128_study_layout import build_bf16_prefill128_layout
from xModel.qwen35.vl_0_8b.kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
from xModel.qwen35.vl_0_8b.bf16_prefill128_kv_profile import validate_bf16_prefill128_kv_program,validate_bf16_prefill128_kv_decoder
from xModel.qwen35.vl_0_8b.bf16_low_batch_kv_profile import validate_bf16_low_batch_kv_program,validate_bf16_low_batch_kv_decoder
from xModel.qwen35.vl_0_8b.bf16_float_reference_profile import validate_bf16_float_reference_program
from xModel.qwen35.vl_0_8b.joined_mlp_profile import validate_joined_mlp_profile
from xModel.qwen38.vl_27b.model import create_cuda_bf16_prefill128_study_decoder

def options():return dict(external_compressed_weights=True,joined_mlp_projections=True)
def decoder_options():return dict(low_batch_decode=True,retain_linear_inputs=True,deferred_state_commit=True,shared_output_storage=True,host_embedding_lookup=True,**options())

def trace128():
    trace=Trace();ns={'T':trace};source=(MODEL/'cuda_packed_program.py').read_bytes()
    functions(source,('packed_view','concat','flat_concat'),ns);functions((MODEL/'cuda_chunk.py').read_bytes(),('Qwen35CUDAChunk',),ns)
    delta=build_bf16_prefill128_layout()
    p=types.SimpleNamespace(config=CONFIG,batch=1,size=128,hidden_width=655360,state_width=106823680,
        layout=copy.deepcopy(delta['layers']),flat_concat=True,delta_layout=delta,bf16_kv_state=True,bf16_kv_low_batch_state=False,bf16_kv_prefill128_state=True,
        external_compressed_weights=True,workspace=0,retain=False,register_delta_128=True,join_input_projections=False,
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
    member(source,'CUDAPackedDecoderProgram','_forward',ns)(p)(*inputs,*weights)
    return trace,p

class Controls(unittest.TestCase):
    def test_distinct_exact_layout_and_unchanged_old_guards(self):
        self.assertEqual(build_bf16_prefill128_layout(),independent_layout())
        validate_bf16_prefill128_kv_program(True,1,4096,128,False,CONFIG,**options())
        validate_bf16_prefill128_kv_decoder(True,1,4096,128,4,CONFIG,**decoder_options())
        with self.assertRaises(ValueError):validate_bf16_low_batch_kv_program(True,1,4096,128,False,CONFIG,**options())
        with self.assertRaises(ValueError):validate_bf16_low_batch_kv_decoder(True,1,4096,128,4,CONFIG,**{k:v for k,v in decoder_options().items() if k!='low_batch_decode'})
    def test_all_profile_flags_and_dimensions_fail_closed(self):
        for args in ((True,4096,128),(4,4096,128),(1,4095,128),(1,4096,32),(1,4096,128.0)):
            with self.subTest(args=args),self.assertRaises(ValueError):validate_bf16_prefill128_kv_program(True,*args,False,CONFIG,**options())
        for key,value in (('external_compressed_weights',False),('joined_mlp_projections',False),('embedded_projection_weights',True),('legacy_bf16_kv_state',True),('bf16_kv_float_reference',True),('join_input_projections',True),('kv_delta_state',True),('kv_overlay_strategy',0)):
            opts=options();opts[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_bf16_prefill128_kv_program(True,1,4096,128,False,CONFIG,**opts)
        for key in ('low_batch_decode','retain_linear_inputs','deferred_state_commit','shared_output_storage','host_embedding_lookup'):
            opts=decoder_options();opts[key]=False
            with self.subTest(key=key),self.assertRaises(ValueError):validate_bf16_prefill128_kv_decoder(True,1,4096,128,4,CONFIG,**opts)
        bad=copy.deepcopy(CONFIG);bad['rope_parameters']['rope_theta']=10000
        with self.assertRaises(ValueError):validate_bf16_prefill128_kv_program(True,1,4096,128,False,bad,**options())
    def test_full_forward_trace_uses_only_new_attention_dialect(self):
        trace,p=trace128();full=[r for r in trace.rows if r['name']=='qwen35_full_decoder_chunk']
        self.assertEqual(len(full),16)
        for i,row in enumerate(full):
            kw=row['kw'];layer=p.delta_layout['layers'][4*i+3]
            self.assertEqual(kw['canonical_bf16_prefill128_kv_overlay'],1)
            for name in ('canonical_bf16_low_batch_kv_overlay','canonical_bf16_kv_overlay','canonical_kv_overlay'):self.assertNotIn(name,kw)
            self.assertEqual((kw['canonical_state_width'],kw['canonical_key_offset'],kw['canonical_value_offset']),(106823680,layer['offset'],layer['offset']+2097152))
        linear=[r for r in trace.rows if r['name']=='qwen35_linear_decoder_chunk'];self.assertEqual(len(linear),48)
        self.assertTrue(all(r['kw']['register_delta_128']==1 and r['kw']['retain_linear_inputs']==0 for r in linear))
        self.assertEqual(sum(r['name']=='qwen35_f32_barrier' for r in trace.rows),63)
        self.assertFalse(any(r['name']=='qwen35_packed_view' and r['kw']['shape']==[1,4,4096,256] for r in trace.rows))
        self.assertEqual(p.delta_layout['output_width'],44564480)
    def test_program_admission_and_exclusivity_before_tensor_request(self):
        class ReachedTensor(Exception):pass
        tensor=types.SimpleNamespace(input=lambda *a,**kw:(_ for _ in ()).throw(ReachedTensor()),float32='f32')
        ns=dict(validate_bf16_prefill128_kv_program=validate_bf16_prefill128_kv_program,validate_bf16_float_reference_program=validate_bf16_float_reference_program,
                build_bf16_prefill128_layout=build_bf16_prefill128_layout,tensor=tensor)
        ctor=member((MODEL/'cuda_packed_program.py').read_bytes(),'CUDAPackedDecoderProgram','__init__',ns)
        obj=types.SimpleNamespace();opts=dict(batch=1,capacity=4096,chunk_size=128,retain_linear_inputs=False,bf16_kv_prefill128_state=True,**options())
        with self.assertRaises(ReachedTensor):ctor(obj,'unused',CONFIG,None,**opts)
        self.assertEqual(obj.delta_layout,independent_layout());self.assertTrue(obj.bf16_kv_prefill128_state)
        for fault in ({'bf16_kv_low_batch_state':True},{'bf16_kv_state':True},{'bf16_kv_prefill128_state':1},{'retain_linear_inputs':True}):
            with self.subTest(fault=fault),self.assertRaises(ValueError):ctor(types.SimpleNamespace(),'unused',CONFIG,None,**dict(opts,**fault))
    def test_missing_new_native_factory_fails_before_io(self):
        touched=[]
        class Native:
            cuda_bf16_low_batch_kv_delta_commit=object()
            def __getattr__(self,name):touched.append(name);raise AttributeError(name)
        ns=dict(validate_bf16_prefill128_kv_decoder=validate_bf16_prefill128_kv_decoder,validate_bf16_low_batch_kv_decoder=validate_bf16_low_batch_kv_decoder,garnet=Native())
        ctor=member((MODEL/'cuda_packed_decoder.py').read_bytes(),'CUDAPackedDecoder','__init__',ns)
        opts=dict(batch=1,capacity=4096,prefill_chunk=128,decode_chunk=4,bf16_kv_low_batch_state=True,bf16_kv_prefill128_study=True,**{k:v for k,v in decoder_options().items() if k!='low_batch_decode'})
        with self.assertRaisesRegex(ValueError,'matching experimental native build'):ctor(types.SimpleNamespace(),'unused',CONFIG,**opts)
        self.assertEqual(touched,['cuda_bf16_prefill128_kv_delta_commit'])
    def test_separate_factory_routes_profiles_and_rejects_wrong_dialect(self):
        config=json.loads((Path(__file__).with_name('evidence')/'nvfp4-config.json').read_bytes());calls=[]
        pool={'compressed_projection_pool_entries':336,'compressed_nvfp4_entries':128,'compressed_fp8_entries':208,'compressed_projection_weight_bytes':100,'compressed_projection_scale_bytes':50,'embedding_mode':'host_mapped_bf16','embedding_table_bytes':0,'host_embedding_mapped_tensor_bytes':2542796800,'head_precision':'modelopt_nvfp4_w4a4'}
        stats={'native_nvfp4_projection_weights':128,'native_fp8_projection_weights':208,'external_bf16_weight_inputs':0,'external_compressed_weight_inputs':336,'external_compressed_weight_bytes':150}
        programs={t:types.SimpleNamespace(batch=1,delta_layout=build_bf16_prefill128_layout() if t==128 else build_bf16_low_batch_layout(chunk=4),graph=types.SimpleNamespace(stats_json=lambda:json.dumps(stats))) for t in (4,128)}
        decoder=types.SimpleNamespace(programs=programs,io=types.SimpleNamespace(enable_nvfp4_head=lambda value:calls.append(('head',value)),memory_json=lambda:json.dumps(pool)),close=lambda:calls.append(('close',)))
        module=types.ModuleType('cpu_T128_decoder_fixture');module.CUDAPackedDecoder=lambda *a,**kw:(calls.append(('construct',a,kw)) or decoder)
        with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}):self.assertIs(create_cuda_bf16_prefill128_study_decoder('fixture',config),decoder)
        self.assertEqual(calls[0][1],('fixture',config['text_config'],1,4096,128,4));self.assertIs(calls[0][2]['bf16_kv_prefill128_study'],True)
        programs[128].delta_layout['contract']=programs[4].delta_layout['contract']
        with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}),self.assertRaises(ValueError):create_cuda_bf16_prefill128_study_decoder('fixture',config)
        self.assertEqual(calls[-1],('close',))
    def test_qualified_cuda_sources_are_exact_copies(self):
        for name in ('qwen38_bf16_prefill128_attention.cu','qwen38_bf16_prefill128_attention.cuh','qwen38_bf16_prefill128_commit.cu','qwen38_bf16_prefill128_commit.cuh'):
            self.assertEqual((ROOT/'src/cuda'/name).read_bytes(),Path(__file__).with_name(name).read_bytes())
        native=(ROOT/'src/backends/tensorrt/qwen38_bf16_prefill128_kv_overlay_plugin.cpp').read_text()
        self.assertIn('a.tokens==128',native);self.assertIn('config_.value_offset,1,stream',native)
        old=(ROOT/'src/backends/tensorrt/qwen38_bf16_low_batch_kv_overlay_plugin.cpp').read_text();self.assertIn('(a.tokens==4||a.tokens==32)',old)

    def test_recorded_native_and_python_source_transformations(self):
        e=Path(__file__).with_name('evidence');sha=lambda raw:hashlib.sha256(raw).hexdigest()
        for name in ('prefill128-native-runtime-derivation.json','prefill128-python-runtime-derivation.json'):
            for row in json.loads((e/name).read_bytes())['files']:
                source=(ROOT/row['source']).read_bytes();actual=(ROOT/row['target']).read_bytes()
                self.assertEqual(sha(source),row['source_sha256']);self.assertEqual(sha(actual),row['target_sha256'])
                if 'replacements' in row:
                    text=source.decode()
                    for step in row['replacements']:
                        self.assertEqual(text.count(step['old']),step['count']);text=text.replace(step['old'],step['new'])
                    self.assertEqual(text.encode(),actual)

if __name__=='__main__':unittest.main()
