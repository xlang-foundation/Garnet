"""Fresh embedded process per B1-reference/B4-candidate phase; fixed owner cache."""
from pathlib import Path
import json,os,subprocess,sys
root,out,cache=map(Path,sys.argv[1:4]);mode=sys.argv[4];reference=Path(sys.argv[5]);assert mode in ('reference','candidate');m=root/'Garnet/test2026/cuda/qwen38_5090'
p=json.loads((m/'evidence/connected-fa-value-preparation.json').read_bytes());env=dict(os.environ);env.update(p['flags']);env['GARNET_FP8_BATCH4_PREFILL_CANONICAL']='0' if mode=='reference' else '1';env['GARNET_FP8_GDN_BATCH4_DECODE_CANONICAL']='0' if mode=='reference' else '1';env.update(GARNET_RUNTIME_ROOT=str(root/'build/bin'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',TMP=str(cache),TEMP=str(cache))
for n in ('GARNET_PROFILE_DECODE_LAYERS','GARNET_MODELOPT_INIT_TRACE','GARNET_Z128_RECONSTRUCT_TRACE','GARNET_Z128_FACTORY_TRACE','GARNET_KV128_FACTORY_TRACE','GARNET_Q128_FACTORY_TRACE','CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_MASK','CUBLASLT_LOG_FILE'):env.pop(n,None)
env['GARNET_GDNM16_FACTORY_TRACE']='1' if mode=='candidate' else '0';env['GARNET_MODELOPT_INIT_TRACE']='1' if mode=='candidate' else '0'
env['PATH']=';'.join((str(root/'build/bin'),str(root/'tools/TensorRT/bin'),str(root/'tools/cuda13/nvidia/cu13/bin/x86_64'),env['PATH']))
raise SystemExit(subprocess.run([str(root/'build/bin/xlang3.exe'),str(m/'probe_connected_fa_value.py'),str(root),str(out),mode,str(reference)],env=env,cwd=root/'Garnet').returncode)
