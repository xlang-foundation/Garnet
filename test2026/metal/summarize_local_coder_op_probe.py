"""Summarize serialized native-op diagnostics, never model serving throughput."""
import argparse,collections,hashlib,json,math
from pathlib import Path
from audit_local_coder_batch_study import audit

def summarize(root):
    proof=audit(root);issues=list(proof['issues']);manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('environment_overrides',{}).get('GARNET_METAL_PROFILE_OPS')!='1':issues.append('Frozen operator profiling mode not enabled')
    groups=collections.defaultdict(list);projection_rows=collections.defaultdict(list)
    for line in (root/'run.log').read_text().splitlines():
        if not line.startswith('metal_op_profile '):continue
        row=json.loads(line.split(' ',1)[1]);ms=row['gpu_ms']
        if not row.get('serialized_diagnostic') or not math.isfinite(ms) or ms<=0:issues.append('Invalid serialized GPU diagnostic')
        groups[row['kind']].append(ms)
        if row['kind']=='MatMul':projection_rows[row['a_rows']].append(ms)
    if not groups:issues.append('No native serialized operation records')
    total=sum(sum(v) for v in groups.values())
    kinds=[dict(kind=k,calls=len(v),gpu_ms=sum(v),share=sum(v)/total if total else None) for k,v in groups.items()]
    kinds.sort(key=lambda row:row['gpu_ms'],reverse=True)
    return dict(status='passed' if not issues else 'failed',issues=issues,scope=__doc__,
        caveat='Fusions disabled and submissions serialized; sums are attribution diagnostics, not TTFT, decode speed or serving latency',
        native_proof=proof,kinds=kinds,matmul_by_input_rows=[dict(rows=k,calls=len(v),gpu_ms=sum(v)) for k,v in sorted(projection_rows.items())],total_serialized_gpu_ms=total,log_sha256=hashlib.sha256((root/'run.log').read_bytes()).hexdigest())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=summarize(a.root)
    with a.output.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
    if r['issues']:raise SystemExit(1)
if __name__=='__main__':main()
