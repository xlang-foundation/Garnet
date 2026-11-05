"""Reject false finite batch-study completion, tokens, memory and cleanup."""
import json,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_study import audit
root=Path(sys.argv[1]).resolve()
assert audit(root)['status']=='passed'
for mutation in ('tokens','repeat','memory','cleanup','exit'):
    with tempfile.TemporaryDirectory() as folder:
        target=Path(folder)
        for item in root.iterdir():
            if item.name in ('result.json','process.json'):target.joinpath(item.name).write_bytes(item.read_bytes())
            else:target.joinpath(item.name).symlink_to(item)
        result=json.loads((target/'result.json').read_text())
        if mutation=='tokens':result['runs'][0]['generated_ids'][0][0]+=1
        elif mutation=='repeat':result['runs'].pop()
        elif mutation=='memory':result['memory_after_prefill_compile']['metal_allocated_bytes']=2**40
        elif mutation=='cleanup':result['memory_after_close_quiescence']['metal_allocated_bytes']=2**30
        else:
            process=json.loads((target/'process.json').read_text());process['returncode']=1
            (target/'process.json').write_text(json.dumps(process))
        (target/'result.json').write_text(json.dumps(result))
        report=audit(target)
        assert report['status']=='failed' and report['issues'],mutation
        print('batch study audit rejects',mutation)
print('batch study audit PASS baseline and five tamper cases')
