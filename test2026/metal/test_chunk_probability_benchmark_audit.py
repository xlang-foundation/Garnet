"""CPU-only corruption checks for the isolated attention comparison archive."""
import json,shutil,sys,tempfile
from pathlib import Path
from run_chunk_probability_benchmark import audit
root=Path(sys.argv[1]);assert audit(root)['status']=='passed'
mutations=('binary','native_source','cohort','missing_cell','negative_time','precision','false_bitwise','negative_delta')
for mutation in mutations:
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'study';shutil.copytree(root,path)
        if mutation in ('binary','native_source'):
            name='garnet_chunk_probability_benchmark' if mutation=='binary' else 'native-source.mm'
            with (path/name).open('ab') as stream:stream.write(b'x')
        elif mutation=='cohort':
            report=json.loads((path/'status.json').read_text());report['runs'].pop();(path/'status.json').write_text(json.dumps(report))
        else:
            raw=path/'cohort-1.jsonl';rows=[json.loads(line) for line in raw.read_text().splitlines()]
            if mutation=='missing_cell':rows.pop()
            elif mutation=='negative_time':rows[1]['candidate_gpu_ms'][0]=-1
            elif mutation=='precision':rows[0]['dtype']='float16'
            elif mutation=='false_bitwise':
                changed=next(r for r in rows if r.get('maximum_control_delta',0)>0);changed['bitwise_equal']=True
            elif mutation=='negative_delta':rows[1]['maximum_control_delta']=-1
            raw.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        result=audit(path);assert result['status']=='failed' and result['issues'],mutation
        print('chunk_probability_audit rejects',mutation)
print('chunk_probability_audit PASS baseline and',len(mutations),'tamper cases; strict bitwise failures remain explicit')
