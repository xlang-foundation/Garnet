"""Launch the private native worker with the pinned source91 study flags."""
from pathlib import Path
import json,os,subprocess,sys
root,cache=map(Path,sys.argv[1:3]);m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/b4-resident-serial-quality-preparation.json').read_bytes());env=dict(os.environ);env.update(p['flags'])
env.update(GARNET_RUNTIME_ROOT=str(root/'build/bin'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',TMP=str(cache),TEMP=str(cache))
for n in ('GARNET_PROFILE_DECODE_LAYERS','GARNET_MODELOPT_INIT_TRACE','GARNET_Z128_RECONSTRUCT_TRACE','GARNET_Z128_FACTORY_TRACE','GARNET_KV128_FACTORY_TRACE','GARNET_Q128_FACTORY_TRACE','GARNET_GDNM16_FACTORY_TRACE','GARNET_VM16_FACTORY_TRACE','GARNET_QKM16_FACTORY_TRACE','CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_MASK','CUBLASLT_LOG_FILE'):env.pop(n,None)
env['PATH']=';'.join((str(root/'build/bin'),str(root/'tools/TensorRT/bin'),str(root/'tools/cuda13/nvidia/cu13/bin/x86_64'),env['PATH']))
raise SystemExit(subprocess.run([str(root/'build/bin/xlang3.exe'),str(m/'native_b4_resident_serial_http_worker.py'),str(root)],cwd=root/'Garnet',env=env).returncode)
