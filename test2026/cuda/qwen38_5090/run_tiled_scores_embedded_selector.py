"""Fresh actual embedded processes; freeze/default/malformed native API controls."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess
def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    assert not args.output.exists();args.output.mkdir();flag='GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES';rows=[]
    for i,value in enumerate((None,'','0','1','true','01',' 1','1 ','2','-1')):
        env=dict(os.environ);env.pop(flag,None)
        if value is not None:env[flag]=value
        env['GARNET_RUNTIME_ROOT']=str(args.root/'build/bin')
        env['PATH']=';'.join(str(args.root/p) for p in ('build/bin','tools/TensorRT/bin','tools/cuda13/nvidia/cu13/bin/x86_64'))+';'+env['PATH']
        command=[str(args.root/'build/bin/xlang3.exe'),str(Path(__file__).with_name('probe_tiled_scores_embedded_selector.py')),str(args.root)]
        r=subprocess.run(command,cwd=args.root/'Garnet',env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        (args.output/(str(i)+'-stdout.txt')).write_bytes(r.stdout);(args.output/(str(i)+'-stderr.txt')).write_bytes(r.stderr)
        assert r.returncode==0,(value,r.returncode,r.stderr.decode(errors='replace'))
        q=json.loads(r.stdout);assert type(q['first']) is bool and q['first']==q['after_environment_flip']==(value=='1') and q['gpu_executed'] is False
        rows.append({'environment_value':value,'exit_code':r.returncode,'actual':q})
    result={'status':'embedded_native_tiled_selector_exact','library_sha256':hashlib.sha256((args.root/'build/bin/garnet.dll').read_bytes()).hexdigest(),'cases':rows,'gpu_executed':False,'performance_admitted':False}
    (args.output/'report.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result));return 0
if __name__=='__main__':raise SystemExit(main())
