"""Launch exactly one embedded CPU or native scalar-IO correctness stage."""
from pathlib import Path
import json,os,subprocess,sys
root=Path(sys.argv[1]);mode=sys.argv[2];m=root/'Garnet/test2026/cuda/qwen38_5090'
assert root==Path('D:/CantorAI/5090-study-20261023') and mode in ('embedded-cpu','resident-io')
env=dict(os.environ);env.update(GARNET_RUNTIME_ROOT=str(root/'build/bin'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0')
env['PATH']=';'.join((str(root/'build/bin'),str(root/'tools/TensorRT/bin'),str(root/'tools/cuda13/nvidia/cu13/bin/x86_64'),env['PATH']))
if mode=='embedded-cpu':argv=[str(m/'run_b4_resident_serial_embedded_cpu.py'),str(root)]
else:
    target=Path(sys.argv[3]);assert target.parent.exists() and not target.exists()
    argv=[str(m/'probe_b4_host_resident_controls_v2.py'),str(root/'models/Qwen3.8-27B-NVFP4'),str(target)]
raise SystemExit(subprocess.run([str(root/'build/bin/xlang3.exe'),*argv],cwd=root/'Garnet',env=env).returncode)
