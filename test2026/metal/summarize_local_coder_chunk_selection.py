"""Audit a fixed-kernel chunk-size probe sweep; retain first and repeat times.

Loaded native probes exclude initialization and HTTP. They are not the repeated
serving campaign and do not establish a speedup against reference engines.
"""
import argparse,hashlib,json,math
from pathlib import Path
from audit_local_coder_batch_study import audit

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def summarize(root):
    plan=json.loads((root/'plan.json').read_text());status=json.loads((root/'status.json').read_text());issues=[];rows=[];common={};inputs={}
    expected={f'{kind}-chunk{size}' for kind in ('short','long') for size in (8,16,32,64)}
    if {case['id'] for case in plan['plans']}!=expected or len(plan['plans'])!=8:issues.append('Chunk study coverage differs')
    for case in plan['plans']:
        study_path=Path(case['study']);study=json.loads(study_path.read_text());name=case['id'];folder=root/name
        if digest(study_path)!=case['sha256']:issues.append(name+': planned study changed')
        normalized=dict(study);normalized.pop('prefill_chunk');regime=name.split('-')[0]
        if regime not in inputs:inputs[regime]=normalized
        elif inputs[regime]!=normalized:issues.append(name+': inputs or non-chunk policy differs')
        if not (folder/'process.json').exists() or json.loads((folder/'process.json').read_text()).get('status')!='exited':continue
        proof=audit(folder)
        if proof['status']!='passed':issues.append(name+': finite learned proof failed')
        manifest=json.loads((folder/'manifest.json').read_text())
        for artifact in ('libgarnet.dylib','CMakeCache.txt','factory-source.py','decoder-source.py','run-source.py'):
            wanted=manifest['files'][artifact]
            if artifact not in common:common[artifact]=wanted
            elif common[artifact]!=wanted:issues.append(name+': fixed implementation changed: '+artifact)
        if json.loads((folder/'study.json').read_text())!=study:issues.append(name+': executed fixture differs')
        result=json.loads((folder/'result.json').read_text());times=[run['prefill_seconds'] for run in result['runs']]
        if len(times)!=2 or any(not math.isfinite(v) or v<=0 for v in times):issues.append(name+': first/repeat timing missing')
        rows.append(dict(id=name,chunk=study['prefill_chunk'],batch=study['batch'],packed_rows=study['prefill_chunk']*study['batch'],
            input_tokens=[len(case['token_ids']) for case in study['requests']],output_tokens_per_row=study['output_tokens'],
            first_prefill_seconds=times[0] if times else None,repeat_prefill_seconds=times[1] if len(times)>1 else None,
            memory_bytes=proof['metal_bytes'],closed_memory_bytes=proof['closed_metal_bytes'],exact_token_status=proof['status']))
    terminal=status['status']=='collected_pending_chunk_selection_review'
    if terminal and len(rows)!=8:issues.append('Completed collection lacks all eight probes')
    if status['status']=='failed':issues.append('Collector failed: '+status.get('error',''))
    return dict(status='failed' if issues else 'passed' if terminal else 'in_progress',issues=issues,scope=__doc__,
        limitations='Two deterministic native repetitions per cell, not three fresh-process serving cohorts. No reference speed ratio.',
        completed_cells=len(rows),current=status.get('current'),rows=rows,fixed_artifact_hashes=common,plan_sha256=digest(root/'plan.json'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=summarize(a.root)
    with a.output.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
    if r['issues']:raise SystemExit(1)
if __name__=='__main__':main()
