"""Validate actual SGLang server_info for the prepared ordinary/NEXTN candidates."""
CHECKPOINT='/mnt/d/CantorAI/5090-study-20261023/models/Qwen3.8-27B-NVFP4'

def validate(info,mode):
 if mode not in ('ordinary','mtp') or not isinstance(info,dict):raise ValueError('Actual SGLang configuration required')
 wanted={'version':'0.5.22','model_path':CHECKPOINT,'served_model_name':'qwen38-27b-nvfp4','context_length':4096,'dtype':'bfloat16','kv_cache_dtype':'bfloat16','mamba_ssm_dtype':'float32','max_running_requests':8,'max_total_tokens':32768,'max_mamba_cache_size':8,'chunked_prefill_size':4096,'disable_radix_cache':True,'disable_flashinfer_autotune':True,'cuda_graph_backend_decode':'full','cuda_graph_bs_decode':[1,2,4,8],'cuda_graph_backend_prefill':'disabled','mm_preprocess_cache_size_mb':0,'enable_prefix_mm_cache':False,'enable_mm_global_cache':False,'speculative_algorithm':None if mode=='ordinary' else 'EAGLE'}
 # Installed speculative_hook resolves NEXTN to EAGLE for this architecture.
 if mode=='mtp':wanted.update(speculative_draft_model_path=CHECKPOINT,speculative_num_steps=3,speculative_eagle_topk=1,speculative_num_draft_tokens=4)
 for key,value in wanted.items():
  if key not in info or type(info[key]) is not type(value) or info[key]!=value:raise ValueError('SGLang setting differs: '+key+'; expected '+repr(value)+' got '+repr(info.get(key)))
 return {'engine':'sglang','generation_mode':mode,'verified_settings':wanted,'context_capacity':4096,'image_cache_policy':'preprocess0_prefixMMoff_globalMMoff','checkpoint_revision':'482ca0f3832238542f8f5295dde86b5f22711d80'}
