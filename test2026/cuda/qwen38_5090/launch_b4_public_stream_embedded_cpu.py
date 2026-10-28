"""Set the native DLL search path, then run CPU-only protocol fixtures."""
from pathlib import Path
import os,subprocess,sys
root=Path(sys.argv[1]);m=root/'Garnet/test2026/cuda/qwen38_5090';env=dict(os.environ)
env.update(GARNET_RUNTIME_ROOT=str(root/'build/bin'),PYTHONDONTWRITEBYTECODE='1')
env['PATH']=';'.join((str(root/'build/bin'),str(root/'tools/TensorRT/bin'),str(root/'tools/cuda13/nvidia/cu13/bin/x86_64'),env['PATH']))
raise SystemExit(subprocess.run([str(root/'build/bin/xlang3.exe'),str(m/'run_b4_public_stream_embedded_cpu.py'),str(root)],cwd=root/'Garnet',env=env).returncode)
