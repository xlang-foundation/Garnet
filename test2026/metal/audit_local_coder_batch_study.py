"""Check frozen finite ragged-prefill/combined-decode evidence after native exit.

This verifies token correctness and sampled memory, not serving performance.
"""
import argparse,hashlib,json,math
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(root):
    issues=[]
    manifest=json.loads((root/'manifest.json').read_text())
    for name,wanted in manifest['files'].items():
        if digest(root/name)!=wanted:issues.append('Frozen artifact differs: '+name)
    study=json.loads((root/'study.json').read_text());result=json.loads((root/'result.json').read_text())
    process=json.loads((root/'process.json').read_text())
    if process.get('status')!='exited' or process.get('returncode')!=0:issues.append('Native process did not exit successfully')
    if result.get('status')!='batch_study_passed_not_full_qualification':issues.append('Study did not complete')
    for name in ('capacity','batch','prefill_chunk','prefill_mode'):
        if result.get(name)!=study.get(name):issues.append('Frozen profile mismatch: '+name)
    oracle=Path(study['oracle_log'])
    if digest(oracle)!=study['oracle_log_sha256']:issues.append('Single-request oracle log changed')
    # Original oracle is line-wrapped native JSON; match full prompt identity,
    # then compare the retained output prefix without relying on task order.
    candidates=[]
    for line in oracle.read_text().splitlines():
        row=json.loads(line)
        if 'line' in row:
            try:row=json.loads(row['line'])
            except json.JSONDecodeError:continue
        if isinstance(row,dict) and row.get('prompt_ids') and row.get('generated_ids'):candidates.append(row)
    for case in study['requests']:
        if not any(row['prompt_ids']==case['token_ids'] and row['generated_ids'][:study['output_tokens']]==case['expected_ids'] for row in candidates):
            issues.append('Frozen case does not match single-request oracle: '+case['id'])
    runs=result.get('runs',[])
    if len(runs)!=2 or [r['repeat'] for r in runs]!=[0,1]:issues.append('Two reset repeats required')
    for run in runs:
        if run['generated_ids']!=[case['expected_ids'] for case in study['requests']]:issues.append('Generated tokens differ from frozen single-request oracle')
        if any(run['scheduler'].get(k)!=0 for k in ('active','tracked')):issues.append('Scheduler retained requests')
        for field in ('prefill_seconds','decode_window_seconds'):
            if not math.isfinite(run[field]) or run[field]<=0:issues.append('Invalid timing diagnostic')
    memories=[v for k,v in result.items() if k.startswith('memory_')]+[r['memory'] for r in runs]
    for memory in memories:
        if memory['metal_allocated_bytes']>memory['metal_recommended_working_set_bytes']-256*1024*1024:issues.append('Sampled Metal workspace exceeded admission limit')
    closed=result.get('memory_after_close_quiescence',{}).get('metal_allocated_bytes')
    if closed is None or closed>1024*1024:issues.append('Native allocations did not return to bounded close baseline')
    return dict(status='passed' if not issues else 'failed',issues=issues,scope=__doc__,
        batch=study['batch'],capacity=study['capacity'],chunk=study['prefill_chunk'],repeats=len(runs),
        metal_bytes=result.get('memory_after_prefill_compile',{}).get('metal_allocated_bytes'),closed_metal_bytes=closed,
        result_sha256=digest(root/'result.json'),manifest_sha256=digest(root/'manifest.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=audit(args.root)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
