"""Embedded native API selection compatibility; no CUDA/model call."""
from pathlib import Path
import json,os,sys
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'build/bin'));sys.path.insert(0,str(root/'Garnet'))
import garnet
flag='GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES'
first=garnet.bf16_low_batch_tiled_scores_enabled();assert type(first) is bool
os.environ[flag]='0' if first else '1'
second=garnet.bf16_low_batch_tiled_scores_enabled();assert type(second) is bool and first==second
print(json.dumps({'first':first,'after_environment_flip':second,'gpu_executed':False}),flush=True)
