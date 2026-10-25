"""Actual DLL route selection/freeze tests only; no CUDA API or model execution."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,subprocess,sys

FLAG='GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES'
def child(root):
    handles=[os.add_dll_directory(str(root/p)) for p in ('build/bin','tools/TensorRT/bin','tools/cuda13/nvidia/cu13/bin/x86_64')]
    library=ctypes.CDLL(str(root/'build/bin/garnet.dll'))
    query=library.GarnetQwen38BF16LowBatchTiledScoresEnabled
    query.argtypes=[];query.restype=ctypes.c_int
    first=query();os.environ[FLAG]='0' if first else '1'
    assert query()==first
    print(json.dumps({'first':first,'after_environment_flip':query(),'gpu_executed':False}))

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path);a.add_argument('--child',action='store_true');args=a.parse_args()
    if args.child:child(args.root);return 0
    assert args.output is not None and not args.output.exists();args.output.mkdir()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    library=args.root/'build/bin/garnet.dll';before=sha(library);cases=[]
    for i,value in enumerate((None,'','0','1','true','01',' 1','1 ','2','-1')):
        env=dict(os.environ);env.pop(FLAG,None)
        if value is not None:env[FLAG]=value
        command=[sys.executable,str(Path(__file__).resolve()),'--root',str(args.root),'--child']
        r=subprocess.run(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (args.output/(str(i)+'-stdout.txt')).write_bytes(r.stdout);(args.output/(str(i)+'-stderr.txt')).write_bytes(r.stderr)
        assert r.returncode==0,(value,r.returncode,r.stderr.decode(errors='replace'))
        q=json.loads(r.stdout);assert q['first']==q['after_environment_flip']==int(value=='1') and q['gpu_executed'] is False
        cases.append({'environment_value':value,'exit_code':r.returncode,'actual':q})
    assert sha(library)==before
    result={'status':'native_tiled_selector_cpu_exact','library_sha256':before,'cases':cases,'gpu_executed':False,'performance_admitted':False}
    (args.output/'report.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0

if __name__=='__main__':raise SystemExit(main())
