"""Freeze and measure native projection variants strictly one process at a time.

Example: --variant base=/build/base --variant tall=/build/tall
         --variant k64=/build/k64 --output /absolute/new-study
"""
import argparse,hashlib,json,re,shutil,subprocess,time
from pathlib import Path
from run_local_coder_baseline import engine_processes

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',action='append',required=True,help='LABEL=BUILD_DIRECTORY; CLI order is measured order')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cohorts',type=int,default=3)
    parser.add_argument('--warmups',type=int,default=3);parser.add_argument('--repetitions',type=int,default=9)
    parser.add_argument('--maximum-rows',type=int,choices=(64,128),default=64,help='128 covers batch2 x chunk64 packed prefill')
    args=parser.parse_args()
    if not 1<=args.cohorts<=10 or not 1<=args.warmups<=20 or not 3<=args.repetitions<=100:parser.error('Invalid cohort/warmup/repetition bounds')
    variants={}
    for value in args.variant:
        label,separator,folder=value.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]{1,32}',label) or label in variants:parser.error('Unique simple labels and explicit build directories required')
        folder=Path(folder).resolve()
        if not (folder/'garnet_affine_int4_benchmark').is_file() or not (folder/'CMakeCache.txt').is_file():parser.error('Missing native executable or cache: '+str(folder))
        variants[label]=folder
    live=engine_processes()
    if live:raise RuntimeError('An inference engine is live: '+str(live))
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('affine_int4_benchmark.cpp');repo=source.parents[2]
    manifest=dict(scope='Synthetic projection GPU/wall latency, explicit operand profiles per variant; excludes loading/compilation/downloads, not model quality or throughput',
        maximum_rows=args.maximum_rows,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),warmups=args.warmups,repetitions=args.repetitions,cohorts=args.cohorts,
        variant_order=list(variants),variants={},required_mpp_variants=[])
    for label,folder in variants.items():
        target=out/label;target.mkdir()
        for name in ('garnet_affine_int4_benchmark','CMakeCache.txt'):shutil.copy2(folder/name,target/name)
        manifest['variants'][label]={p.name:digest(p) for p in target.iterdir()}
        cache=(target/'CMakeCache.txt').read_text()
        if any('GARNET_METAL_MPP_AFFINE_'+kind+':BOOL=ON' in cache for kind in ('FP32','BF16')):manifest['required_mpp_variants'].append(label)
    shutil.copy2(source,out/'benchmark-source.cpp');shutil.copy2(Path(__file__),out/'runner-source.py')
    manifest['source_sha256']=digest(out/'benchmark-source.cpp');manifest['runner_sha256']=digest(out/'runner-source.py')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    status=dict(status='running',runs=[])
    def save():(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    try:
        for cohort in range(1,args.cohorts+1):
            for label in variants:
                live=engine_processes()
                if live:raise RuntimeError('An inference engine is live: '+str(live))
                began=time.monotonic()
                with (out/f'{label}-cohort{cohort}.jsonl').open('x') as stdout,(out/f'{label}-cohort{cohort}.stderr').open('x') as stderr:
                    process=subprocess.Popen([str(out/label/'garnet_affine_int4_benchmark'),str(args.repetitions),str(args.warmups),str(args.maximum_rows)],stdout=stdout,stderr=stderr)
                    status['current']=dict(variant=label,cohort=cohort,pid=process.pid);save();returncode=process.wait()
                status['runs'].append(dict(variant=label,cohort=cohort,returncode=returncode,total_seconds=time.monotonic()-began));status.pop('current',None);save()
                print(json.dumps(status['runs'][-1]),flush=True)
                if returncode:raise RuntimeError('Native operator failed; raw artifacts retained')
        status['status']='collected_pending_audit';save()
    except Exception as error:status['status']='failed';status['error']=repr(error);save();raise
if __name__=='__main__':main()
