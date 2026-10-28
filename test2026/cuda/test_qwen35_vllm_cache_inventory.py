# SPDX-License-Identifier: Apache-2.0
"""CPU verification of cache view/storage accounting, never a vLLM GPU claim."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from qwen35_vllm_memory import _cache_inventory

assert not torch.cuda.is_initialized()
raw=torch.zeros(64,dtype=torch.uint8)
before=raw.clone()
class Layer:
    kv_cache=(raw[:32].view(torch.bfloat16).reshape(2,8),raw[32:].view(torch.float32).reshape(2,4))
    def get_state_dtype(self):return (torch.bfloat16,torch.float32)
runner=SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context={'gdn':Layer()}),kv_caches=[raw])
first=_cache_inventory(torch,runner,Layer)
assert first['static_forward_context_available'] and first['typed_layer_view_count']==2
assert first['unique_referenced_storage_bytes']==64 and len(first['unique_referenced_storages'])==1
assert [v['dtype'] for v in first['views']]==['torch.bfloat16','torch.float32','torch.uint8']
assert [v['logical_bytes'] for v in first['views']]==[32,32,64]
assert [v['role'] for v in first['views'][:2]]==['gated_delta_conv_state','gated_delta_recurrent_state']
assert first['layers'][0]['bound_dtypes_match_resolved_state_dtypes']
assert len({v['storage_id'] for v in first['views']})==1
runner.kv_caches=[raw,raw.view(2,32),torch.zeros(64,dtype=torch.uint8)]
second=_cache_inventory(torch,runner,Layer)
assert second['unique_referenced_storage_bytes']==128 and len(second['unique_referenced_storages'])==2
assert torch.equal(raw,before)
missing=_cache_inventory(torch,SimpleNamespace())
assert not missing['static_forward_context_available'] and missing['typed_layer_view_count']==0
assert missing['unique_referenced_storage_bytes']==0
assert not torch.cuda.is_initialized()
result={'all_passed':True,'cpu_torch':torch.__version__,'cuda_context_initialized':False,
        'shared_mixed_dtype_storage_bytes':first['unique_referenced_storage_bytes'],
        'shared_mixed_dtype_views':first['views'],'two_distinct_storages_bytes':second['unique_referenced_storage_bytes'],
        'scope':'Actual CPU tensors with BF16/FP32 typed views of one byte allocation. Verifies metadata/deduplication and readonly behavior; not vLLM GPU cache allocation or worker RPC validation.'}
Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8')
print('VLLM_CACHE_INVENTORY_CPU_ACCOUNTING_PASS')
