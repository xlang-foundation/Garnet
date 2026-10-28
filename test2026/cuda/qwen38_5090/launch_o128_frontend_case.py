"""Set native policy before loading Garnet in a fresh embedded process on Windows."""
from pathlib import Path
import os,subprocess,sys
def main():
    root,out,cache=map(Path,sys.argv[1:4]);index=int(sys.argv[4]);sys.path.insert(0,str(root/'Garnet'))
    from test2026.cuda.qwen38_5090.o128_frontend_contract import CASES
    env=dict(os.environ);env.update(CASES[index]['flags']);env.update(GARNET_RUNTIME_ROOT=str(root/'build/bin'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',TMP=str(cache),TEMP=str(cache))
    for n in ('GARNET_PROFILE_DECODE_LAYERS','GARNET_MODELOPT_INIT_TRACE','GARNET_Z128_RECONSTRUCT_TRACE','GARNET_Z128_FACTORY_TRACE','GARNET_KV128_FACTORY_TRACE','GARNET_Q128_FACTORY_TRACE','CUBLASLT_LOG_LEVEL','CUBLASLT_LOG_MASK','CUBLASLT_LOG_FILE'):env.pop(n,None)
    env['PATH']=';'.join((str(root/'build/bin'),str(root/'tools/TensorRT/bin'),str(root/'tools/cuda13/nvidia/cu13/bin/x86_64'),env['PATH']))
    return subprocess.run([str(root/'build/bin/xlang3.exe'),str(root/'Garnet/test2026/cuda/qwen38_5090/probe_o128_frontend.py'),str(root),str(out),str(index)],env=env,cwd=root/'Garnet').returncode
if __name__=='__main__':raise SystemExit(main())
