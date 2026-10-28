"""CPU profile boundaries over the real pinned checkpoint metadata; no GPU imports."""
from pathlib import Path
import copy,hashlib,json,unittest
from .bf16_low_batch_profile import validate_bf16_low_batch_kv_program,validate_bf16_low_batch_kv_decoder
from xModel.qwen35.vl_0_8b.bf16_kv_profile import validate_bf16_kv_program

class LowBatchProfileControls(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  raw=(Path(__file__).with_name('evidence')/'nvfp4-config.json').read_bytes()
  assert hashlib.sha256(raw).hexdigest()=='71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8'
  all_config=json.loads(raw);assert all_config['architectures']==['Qwen3_5ForConditionalGeneration'];cls.config=all_config['text_config']
 def program(self,**changes):
  flags={'enabled':True,'batch':1,'capacity':4096,'chunk':4,'retain':True,'config':copy.deepcopy(self.config),
   'external_compressed_weights':True,'joined_mlp_projections':True}
  flags.update(changes);return validate_bf16_low_batch_kv_program(**flags)
 def decoder(self,**changes):
  flags={'enabled':True,'batch':1,'capacity':4096,'prefill':32,'decode':4,'config':copy.deepcopy(self.config),
   'retain_linear_inputs':True,'deferred_state_commit':True,'shared_output_storage':True,'host_embedding_lookup':True,
   'external_compressed_weights':True,'joined_mlp_projections':True}
  flags.update(changes);return validate_bf16_low_batch_kv_decoder(**flags)
 def test_actual_metadata_positive_and_legacy_cross_rejection(self):
  self.assertIsNone(self.program());self.assertIsNone(self.program(chunk=32,retain=False));self.assertIsNone(self.decoder())
  for chunk in (4,32):
   with self.assertRaises(ValueError):validate_bf16_kv_program(True,1,4096,chunk,chunk==4,self.config,external_compressed_weights=True,joined_mlp_projections=True)
   for batch in (4,8):
    self.assertIsNone(validate_bf16_kv_program(True,batch,4096,chunk,chunk==4,self.config,external_compressed_weights=True,joined_mlp_projections=True))
    with self.assertRaises(ValueError):self.program(batch=batch,chunk=chunk,retain=chunk==4)
 def test_strict_scalar_geometry_and_retain(self):
  for field,values in [('batch',(False,True,0,2,4,8,1.,'1')),('capacity',(4095,4097,False,4096.)),('chunk',(8,16,128,True,4.)),('kv_overlay_strategy',(False,True,0,2,1.,'1'))]:
   for value in values:
    with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.program(**{field:value})
  for chunk in (4,32):
   for retain in (None,0,1,not(chunk==4)):
    with self.subTest(chunk=chunk,retain=retain),self.assertRaises(ValueError):self.program(chunk=chunk,retain=retain)
 def test_mode_and_shared_projection_exclusions(self):
  for field,values in [('legacy_bf16_kv_state',(True,0,1,None)),('bf16_kv_float_reference',(True,0,1,None)),('external_compressed_weights',(False,1,None)),('joined_mlp_projections',(False,1,None)),('embedded_projection_weights',(True,0,None)),('join_input_projections',(True,0,None)),('kv_delta_state',(True,0,None))]:
   for value in values:
    with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.program(**{field:value})
 def test_full_architecture_mutations(self):
  for field in ('hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim','linear_conv_kernel_dim'):
   for value in (self.config[field]+1,True,float(self.config[field])):
    cfg=copy.deepcopy(self.config);cfg[field]=value
    with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.program(config=cfg)
  configs=[]
  for layers in (self.config['layer_types'][:-1],self.config['layer_types']+[self.config['layer_types'][0]],list(reversed(self.config['layer_types']))):
   cfg=copy.deepcopy(self.config);cfg['layer_types']=layers;configs.append(cfg)
  cfg=copy.deepcopy(self.config);cfg['rms_norm_eps']=1e-5;configs.append(cfg)
  for field,value in [('partial_rotary_factor',.5),('mrope_section',[10,11,11]),('rope_theta',1000000)]:
   cfg=copy.deepcopy(self.config);cfg['rope_parameters'][field]=value;configs.append(cfg)
  for cfg in configs:
   with self.assertRaises(ValueError):self.program(config=cfg)
 def test_decoder_lifecycle_flags_and_chunk_limits(self):
  for field in ('retain_linear_inputs','deferred_state_commit','shared_output_storage','host_embedding_lookup'):
   for value in (False,0,1,None):
    with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.decoder(**{field:value})
  for field,values in [('prefill',(4,16,128,32.,True)),('decode',(8,16,4.,True))]:
   for value in values:
    with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.decoder(**{field:value})
  for field in ('legacy_bf16_kv_state','bf16_kv_float_reference'):
   with self.assertRaises(ValueError):self.decoder(**{field:True})
 def test_explicit_boolean_optin_and_disabled_semantics(self):
  for enabled in (0,1,None,'1'):
   with self.assertRaises(ValueError):self.program(enabled=enabled)
   with self.assertRaises(ValueError):self.decoder(enabled=enabled)
  self.assertIsNone(self.program(enabled=False,batch='invalid'))
  self.assertIsNone(self.decoder(enabled=False,batch='invalid'))
 def test_foreign_small_model_geometry_rejected(self):
  cfg=copy.deepcopy(self.config);cfg.update(hidden_size=2560,intermediate_size=9216,num_attention_heads=16,num_key_value_heads=4)
  with self.assertRaises(ValueError):self.program(config=cfg)
  with self.assertRaises(ValueError):self.decoder(config=cfg)

if __name__=='__main__':unittest.main()
