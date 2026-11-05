"""Freeze and run a finite learned batch probe, optionally serializing GPU ops.

The operator trace disables some fusions and is never serving performance data.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
from run_local_coder_baseline import engine_processes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime','build','checkpoint','study','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile-ops',action='store_true');args=p.parse_args()
    live=engine_processes()
    if live:raise RuntimeError('An inference engine is live: '+str(live))
    source=Path(__file__).resolve();repo=source.parents[2];out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    artifacts={'study.json':args.study,'run-source.py':source.with_name('run_local_coder_27b_batch_study.py'),
        'collector-source.py':source,'factory-source.py':repo/'xModel/qwen38/vl_27b/metal_batch_study.py',
        'decoder-source.py':repo/'xModel/qwen35/vl_0_8b/metal_decoder.py',
        'libgarnet.dylib':args.build/'libgarnet.dylib','CMakeCache.txt':args.build/'CMakeCache.txt'}
    for name,path in artifacts.items():shutil.copy2(path,out/name)
    env=dict(os.environ);env['GARNET_METAL_PROFILE_OPS']='1' if args.profile_ops else '0'
    manifest=dict(scope=__doc__,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        environment_overrides={'GARNET_METAL_PROFILE_OPS':env['GARNET_METAL_PROFILE_OPS']},
        files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in artifacts})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    command=[str(args.runtime.resolve()),'-B',str(out/'run-source.py'),str(out),str(repo),str(args.checkpoint.resolve()),str(out/'study.json'),str(out/'result.json')]
    with (out/'run.log').open('xb') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        record=dict(status='running',pid=process.pid,command=command)
        (out/'process.json').write_text(json.dumps(record,indent=2)+'\n');code=process.wait()
        record.update(status='exited',returncode=code);(out/'process.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if code:raise SystemExit(code)
if __name__=='__main__':main()
