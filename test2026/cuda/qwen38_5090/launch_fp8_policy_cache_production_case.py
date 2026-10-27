from pathlib import Path
import os,subprocess,sys
def main():
    root,out,cache=map(Path,sys.argv[1:4]);index=int(sys.argv[4]);sys.path.insert(0,str(root/'Garnet'));from test2026.cuda.qwen38_5090.fp8_policy_cache_production_contract import CASES
    env=dict(os.environ)
    for n,v in CASES[index]['environment'].items():
        if v is None:env.pop(n,None)
        else:env[n]=v
    scratch=cache.parent/'scratch';assert scratch.is_dir();env.update(GARNET_RUNTIME_ROOT=str(root/'build/bin'),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',TMP=str(scratch),TEMP=str(scratch));env['PATH']=';'.join((str(root/'build/bin'),str(root/'tools/TensorRT/bin'),str(root/'tools/cuda13/nvidia/cu13/bin/x86_64'),env['PATH']))
    return subprocess.run([str(root/'build/bin/xlang3.exe'),str(root/'Garnet/test2026/cuda/qwen38_5090/probe_fp8_policy_cache_production.py'),str(root),str(out),str(cache),str(index)],env=env,cwd=root/'Garnet').returncode
if __name__=='__main__':raise SystemExit(main())
