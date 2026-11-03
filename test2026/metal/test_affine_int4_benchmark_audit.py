"""CPU-only corruption checks for archived synthetic projection cohorts."""
import json,shutil,sys,tempfile
from pathlib import Path
from summarize_affine_int4_benchmark import summarize
root=Path(sys.argv[1]);assert summarize(root)['status']=='passed'
for mutation in ('negative_gpu_time','missing_cell','missing_mpp','changed_binary','skipped_cohort'):
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'study';shutil.copytree(root,path)
        if mutation=='changed_binary':
            with (path/'mpp/garnet_affine_int4_benchmark').open('ab') as stream:stream.write(b'x')
        elif mutation=='skipped_cohort':
            report=json.loads((path/'status.json').read_text());report['runs'].pop();(path/'status.json').write_text(json.dumps(report))
        else:
            raw=path/'mpp-cohort1.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
            if mutation=='negative_gpu_time':rows[1]['gpu_ms'][0]=-1
            elif mutation=='missing_cell':rows.pop()
            else:rows[0]['mpp_available']=False
            raw.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        result=summarize(path);assert result['status']=='failed' and result['issues'],mutation
        print('affine_int4_benchmark_audit rejects',mutation)
print('affine_int4_benchmark_audit PASS baseline and five tamper cases')
