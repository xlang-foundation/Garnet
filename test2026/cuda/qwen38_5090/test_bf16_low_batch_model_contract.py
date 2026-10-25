"""Local opt-in B1 construction/routing checks; no native runtime or GPU model."""
import ast,copy,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from .test_bf16_kv_model_contract import build_trace
from .test_kv_delta_graph_contract import Trace,functions,member,MODEL,CONFIG
from xModel.qwen35.vl_0_8b.bf16_low_batch_kv_profile import validate_bf16_low_batch_kv_program,validate_bf16_low_batch_kv_decoder
from xModel.qwen35.vl_0_8b.kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
from xModel.qwen35.vl_0_8b.joined_mlp_profile import validate_joined_mlp_profile
from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder

class LowBatchModelContracts(unittest.TestCase):
 def test_distinct_marker_and_no_full_fp32_cache_views(self):
  for chunk in (4,32):
   trace,p=build_trace(1,chunk,low_batch=True);full=[r for r in trace.rows if r['name']=='qwen35_full_decoder_chunk'];self.assertEqual(len(full),16)
   for i,row in enumerate(full):
    kw=row['kw'];layer=p.delta_layout['layers'][4*i+3]
    self.assertEqual(kw['canonical_bf16_low_batch_kv_overlay'],1);self.assertNotIn('canonical_bf16_kv_overlay',kw);self.assertNotIn('canonical_kv_overlay',kw)
    self.assertEqual((kw['canonical_state_width'],kw['canonical_key_offset'],kw['canonical_value_offset']),(106823680,layer['offset'],layer['offset']+2097152))
   views=[r for r in trace.rows if r['name']=='qwen35_packed_view'];self.assertTrue(all(r['kw']['shape'][0]==1 for r in views))
   self.assertFalse(any(r['kw']['shape']==[1,4,4096,256] for r in views))
   linear=[r for r in trace.rows if r['name']=='qwen35_linear_decoder_chunk'];self.assertEqual(len(linear),48)
   self.assertTrue(all(r['kw']['register_delta_128']==1 and r['kw']['retain_linear_inputs']==int(chunk==4) for r in linear))
   self.assertEqual(sum(r['name']=='qwen35_f32_barrier' for r in trace.rows),63)
 def test_low_batch_layout_not_mutated(self):
  for chunk in (4,32):
   _,p=build_trace(1,chunk,low_batch=True);self.assertEqual(p.delta_layout,build_bf16_low_batch_layout(chunk=chunk,retain_linear_inputs=chunk==4))
   self.assertEqual(p.delta_layout['contract'],'garnet_qwen38_bf16_low_batch_delta_study_v1');self.assertIs(p.delta_layout['runtime_qualified'],False)
 def test_joined_projection_modes_remain_separate(self):
  for chunk in (4,32):self.assertIsNone(validate_joined_mlp_profile(True,1,4096,chunk,True,bf16_low_batch_cache_profile=True))
  with self.assertRaises(ValueError):validate_joined_mlp_profile(True,1,4096,32,True)
  for batch,capacity,chunk in ((4,4096,4),(8,4096,32),(1,512,4),(1,4096,128)):
   with self.assertRaises(ValueError):validate_joined_mlp_profile(True,batch,capacity,chunk,True,bf16_low_batch_cache_profile=True)
  for flags in ({'bf16_cache_profile':True},{'bf16_float_reference':True},{'bf16_low_batch_cache_profile':1}):
   options={'bf16_low_batch_cache_profile':True};options.update(flags)
   with self.assertRaises(ValueError):validate_joined_mlp_profile(True,1,4096,4,True,**options)
 def test_bad_program_profiles_fail_before_io(self):
  ctor=member((MODEL/'cuda_packed_program.py').read_bytes(),'CUDAPackedDecoderProgram','__init__',{'validate_bf16_low_batch_kv_program':validate_bf16_low_batch_kv_program})
  for flags in ({'batch':4},{'capacity':512},{'chunk_size':128},{'retain_linear_inputs':False},{'bf16_kv_state':True},{'bf16_kv_float_reference':True},{'bf16_kv_low_batch_state':1}):
   options={'batch':1,'capacity':4096,'chunk_size':4,'retain_linear_inputs':True,'external_compressed_weights':True,'joined_mlp_projections':True,'bf16_kv_low_batch_state':True};options.update(flags)
   with self.assertRaises(ValueError):ctor(types.SimpleNamespace(),'unused',CONFIG,None,**options)
 def test_missing_matching_native_factory_fails_before_tensor_io(self):
  touched=[]
  class OldNative:
   def __getattr__(self,name):touched.append(name);raise AttributeError(name)
  ctor=member((MODEL/'cuda_packed_decoder.py').read_bytes(),'CUDAPackedDecoder','__init__',{'validate_bf16_low_batch_kv_decoder':validate_bf16_low_batch_kv_decoder,'garnet':OldNative()})
  options={'batch':1,'capacity':4096,'prefill_chunk':32,'decode_chunk':4,'retain_linear_inputs':True,'deferred_state_commit':True,'shared_output_storage':True,'host_embedding_lookup':True,'external_compressed_weights':True,'joined_mlp_projections':True,'bf16_kv_low_batch_state':True}
  with self.assertRaisesRegex(ValueError,'matching experimental native build'):ctor(types.SimpleNamespace(),'unused',CONFIG,**options)
  self.assertEqual(touched,['cuda_bf16_low_batch_kv_delta_commit'])
 def test_bad_decoder_profiles_fail_before_any_native_lookup(self):
  ctor=member((MODEL/'cuda_packed_decoder.py').read_bytes(),'CUDAPackedDecoder','__init__',{'validate_bf16_low_batch_kv_decoder':validate_bf16_low_batch_kv_decoder})
  for field,value in [('batch',4),('prefill_chunk',128),('shared_output_storage',False),('host_embedding_lookup',False),('bf16_kv_low_batch_state',1)]:
   options={'batch':1,'capacity':4096,'prefill_chunk':32,'decode_chunk':4,'retain_linear_inputs':True,'deferred_state_commit':True,'shared_output_storage':True,'host_embedding_lookup':True,'external_compressed_weights':True,'joined_mlp_projections':True,'bf16_kv_low_batch_state':True};options[field]=value
   with self.assertRaises(ValueError):ctor(types.SimpleNamespace(),'unused',CONFIG,**options)
 def test_chunk_cache_modes_are_exclusive(self):
  trace=Trace();ns={'T':trace};functions((MODEL/'cuda_chunk.py').read_bytes(),('Qwen35CUDAChunk',),ns);nodes=[trace.input(str(i)) for i in range(7)]
  for flags in ({'bf16_canonical_kv':True},{'bf16_kv_float_reference':True},{'bf16_low_batch_canonical_kv':1},{'canonical_kv_state':None}):
   options={'canonical_kv_state':nodes[-1],'canonical_kv_layout':{},'bf16_low_batch_canonical_kv':True};options.update(flags)
   with self.assertRaises(ValueError):ns['Qwen35CUDAChunk'](nodes[0],None,None,nodes[1],nodes[2],nodes[3],CONFIG,3,'full_attention',**options)
 def test_optin_defaults_false(self):
  for file,cls in [('cuda_packed_decoder.py','CUDAPackedDecoder'),('cuda_packed_program.py','CUDAPackedDecoderProgram')]:
   tree=ast.parse((MODEL/file).read_bytes());method=next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
   values=dict(zip((n.arg for n in method.args.args[-len(method.args.defaults):]),method.args.defaults));self.assertIs(ast.literal_eval(values['bf16_kv_low_batch_state']),False)
 def factory_fixture(self):
  config=json.loads((Path(__file__).with_name('evidence')/'nvfp4-config.json').read_bytes());calls=[]
  pool={'compressed_projection_pool_entries':336,'compressed_nvfp4_entries':128,'compressed_fp8_entries':208,'compressed_projection_weight_bytes':100,'compressed_projection_scale_bytes':50,'embedding_mode':'host_mapped_bf16','embedding_table_bytes':0,'host_embedding_mapped_tensor_bytes':2542796800,'head_precision':'modelopt_nvfp4_w4a4'}
  stats={'native_nvfp4_projection_weights':128,'native_fp8_projection_weights':208,'external_bf16_weight_inputs':0,'external_compressed_weight_inputs':336,'external_compressed_weight_bytes':150}
  programs={t:types.SimpleNamespace(batch=1,delta_layout=build_bf16_low_batch_layout(chunk=t,retain_linear_inputs=t==4),graph=types.SimpleNamespace(stats_json=lambda:json.dumps(stats))) for t in (4,32)}
  decoder=types.SimpleNamespace(programs=programs,io=types.SimpleNamespace(enable_nvfp4_head=lambda v:calls.append(('head',v)),memory_json=lambda:json.dumps(pool)),close=lambda:calls.append(('close',)))
  module=types.ModuleType('test_cpu_decoder_stub');module.CUDAPackedDecoder=lambda *a,**kw:(calls.append(('construct',a,kw)) or decoder)
  return config,calls,pool,stats,decoder,module
 def test_factory_forwards_physical_b1_and_fixed_recipe(self):
  config,calls,pool,stats,decoder,module=self.factory_fixture()
  with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}):self.assertIs(create_cuda_bf16_low_batch_study_decoder('fixture',config),decoder)
  _,args,options=calls[0];self.assertEqual(args,('fixture',config['text_config'],1,4096,32,4))
  self.assertTrue(all(options[n] is True for n in ('bf16_kv_low_batch_state','flat_concat','register_delta_128','retain_linear_inputs','deferred_state_commit','shared_output_storage','host_embedding_lookup','external_compressed_weights','joined_mlp_projections')))
  self.assertEqual(calls[1:], [('head',True)])
 def test_factory_rejects_mismatched_pool_and_head_and_closes(self):
  for key,value in [('compressed_nvfp4_entries',192),('head_precision','bf16'),('embedding_mode','device_bf16')]:
   config,calls,pool,stats,decoder,module=self.factory_fixture();pool[key]=value
   with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}),self.assertRaises(ValueError):create_cuda_bf16_low_batch_study_decoder('fixture',config)
   self.assertEqual(calls[-1],('close',))
 def test_factory_rejects_wrong_graph_or_physical_batch_and_closes(self):
  for which in ('weights','batch','dialect'):
   config,calls,pool,stats,decoder,module=self.factory_fixture()
   if which=='weights':stats['external_bf16_weight_inputs']=1
   if which=='batch':decoder.programs[32].batch=4
   if which=='dialect':decoder.programs[4].delta_layout['contract']='garnet_qwen38_bf16_delta_study_v1'
   with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}),self.assertRaises(ValueError):create_cuda_bf16_low_batch_study_decoder('fixture',config)
   self.assertEqual(calls[-1],('close',))
 def test_factory_rejects_bad_admission_before_constructing(self):
  for options in ({'capacity':512},{'batch':4},{'batch':True}):
   config,calls,pool,stats,decoder,module=self.factory_fixture()
   with patch.dict(sys.modules,{'xModel.qwen35.vl_0_8b.cuda_packed_decoder':module}),self.assertRaises(ValueError):create_cuda_bf16_low_batch_study_decoder('fixture',config,**options)
   self.assertEqual(calls,[])

if __name__=='__main__':unittest.main()
