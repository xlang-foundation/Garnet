"""Isolated attention A/B cohorts; numeric checks are distinct from bit parity.

The native executable alternates control/candidate order within each cell.
It uses the same FP32 operands and original numerical bounds. It does not
establish learned-model correctness or full-model throughput.
"""
import argparse,hashlib,json,math,shutil,statistics,subprocess,time
from pathlib import Path
from run_local_coder_baseline import engine_processes

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
def audit(root):
    manifest=json.loads((root/'manifest.json').read_text());status=json.loads((root/'status.json').read_text());issues=[]
    policy=manifest.get('candidate_policy','probability')
    if policy not in ('probability','index32'):issues.append('Unsupported attention policy')
    for name,wanted in manifest['files'].items():
        if digest(root/name)!=wanted:issues.append('Frozen artifact changed: '+name)
    if status['status']!='collected' or [(r['cohort'],r['returncode']) for r in status['runs']]!=[(i,0) for i in range(1,manifest['cohorts']+1)]:issues.append('Collection/cohort policy differs')
    expected={(b,128,p) for b in (1,2) for p in (0,128,1024,1920,3968)}|{(1,4,256),(1,6,2048)};cells={}
    for cohort in range(1,manifest['cohorts']+1):
        rows=[json.loads(line) for line in (root/f'cohort-{cohort}.jsonl').read_text().splitlines()]
        metadata=[r for r in rows if r.get('kind')=='metadata'];measurements=[r for r in rows if r.get('kind')=='cell']
        if len(metadata)!=1:issues.append('Metadata missing or duplicated');continue
        meta=metadata[0]
        if meta.get('candidate_policy','probability')!=policy:issues.append('Compiled attention policy differs')
        for field,wanted in dict(device='Apple M5',physical_batch=2,chunk=128,capacity=4096,heads=24,kv_heads=4,dim=256,index_bytes=2,dtype='float32',gated=True,order='alternating within each cell').items():
            if meta.get(field)!=wanted:issues.append('Attention geometry/precision/order differs: '+field)
        keys=[(r['active_rows'],r['active_tokens_per_row'],r['prefix']) for r in measurements]
        if len(keys)!=len(expected) or set(keys)!=expected:issues.append('Attention shape coverage differs')
        for row,key in zip(measurements,keys):
            if row['warmups']!=manifest['warmups'] or row['repetitions']!=manifest['repetitions']:issues.append('Sample policy differs')
            delta=row.get('maximum_control_delta');bitwise=row.get('bitwise_equal')
            if type(bitwise) is not bool or type(delta) not in (int,float) or not math.isfinite(delta) or delta<0 or (bitwise and delta!=0):issues.append('Invalid numerical/parity evidence')
            for field in ('control_gpu_ms','candidate_gpu_ms','control_wall_ms','candidate_wall_ms'):
                samples=row[field]
                if len(samples)!=manifest['repetitions'] or any(not math.isfinite(v) or v<=0 for v in samples):issues.append('Invalid timing samples')
            cells.setdefault(key,[]).append(dict(cohort=cohort,control_gpu_ms=statistics.median(row['control_gpu_ms']),candidate_gpu_ms=statistics.median(row['candidate_gpu_ms']),maximum_control_delta=delta,bitwise_equal=bitwise))
    cache=(root/'CMakeCache.txt').read_text()
    flag='GARNET_METAL_CHUNK_ATTENTION_INDEX32' if policy=='index32' else 'GARNET_METAL_CHUNK_PROBABILITY_REUSE'
    other='GARNET_METAL_CHUNK_PROBABILITY_REUSE' if policy=='index32' else 'GARNET_METAL_CHUNK_ATTENTION_INDEX32'
    if flag+':BOOL=ON' not in cache or other+':BOOL=ON' in cache:issues.append('Required isolated candidate pipeline was not compiled')
    results=[]
    for key,values in sorted(cells.items()):
        if len(values)!=manifest['cohorts']:issues.append('Missing shape cohort')
        control=statistics.median(v['control_gpu_ms'] for v in values);candidate=statistics.median(v['candidate_gpu_ms'] for v in values)
        results.append(dict(active_rows=key[0],active_tokens_per_row=key[1],prefix=key[2],control_gpu_ms=control,candidate_gpu_ms=candidate,ratio=control/candidate,
            bitwise_equal=all(v['bitwise_equal'] for v in values),maximum_control_delta=max(v['maximum_control_delta'] for v in values),cohorts=values))
    return dict(status='passed' if not issues else 'failed',issues=issues,scope=__doc__,strict_bitwise_equivalence=all(r['bitwise_equal'] for r in results),cells=results,manifest_sha256=digest(root/'manifest.json'),auditor_sha256=digest(Path(__file__)))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--policy',choices=('probability','index32'),default='probability');args=parser.parse_args()
    if engine_processes():raise RuntimeError('An inference engine is live')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);source=Path(__file__).with_name('chunk_probability_benchmark.cpp');repo=source.parents[2]
    for path,name in [(args.build/'garnet_chunk_probability_benchmark','garnet_chunk_probability_benchmark'),(args.build/'CMakeCache.txt','CMakeCache.txt'),(source,'benchmark-source.cpp'),(Path(__file__),'runner-source.py'),(repo/'src/backends/metal/metal_context.mm','native-source.mm'),(repo/'src/backends/metal/metal_context.h','native-source.h')]:shutil.copy2(path,root/name)
    if args.policy=='index32':shutil.copy2(repo/'src/backends/metal/chunk_attention_index_bounds.h',root/'index-bounds.h')
    manifest=dict(candidate_policy=args.policy,cohorts=3,warmups=3,repetitions=9,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),files={p.name:digest(p) for p in root.iterdir()})
    write(root/'manifest.json',manifest);status=dict(status='running',runs=[])
    def save():(root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    for cohort in range(1,4):
        if engine_processes():raise RuntimeError('A competing inference engine is live')
        began=time.monotonic()
        with (root/f'cohort-{cohort}.jsonl').open('x') as out,(root/f'cohort-{cohort}.stderr').open('x') as err:
            result=subprocess.run([str(root/'garnet_chunk_probability_benchmark'),'9','3'],stdout=out,stderr=err)
        status['runs'].append(dict(cohort=cohort,returncode=result.returncode,seconds=time.monotonic()-began));save();print(json.dumps(status['runs'][-1]),flush=True)
        if result.returncode:status['status']='failed';save();raise SystemExit(result.returncode)
    status['status']='collected';save();report=audit(root);write(root/'audit1.json',report)
    print(json.dumps(dict(status=report['status'],issues=report['issues'],strict_bitwise_equivalence=report['strict_bitwise_equivalence'],cells=len(report['cells']))),flush=True)
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
