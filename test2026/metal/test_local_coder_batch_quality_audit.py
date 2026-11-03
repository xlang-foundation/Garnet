"""CPU tamper checks against an archived concurrent coding study.

Argument: collected study directory. No engine runs or candidate code executes.
"""
import json,shutil,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_quality import audit
root=Path(sys.argv[1]);assert audit(root)['status']=='passed'
for mutation in ('no_combined','reused_epochs','bad_close','bad_text','cached_output'):
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'study';shutil.copytree(root,path)
        log=path/'server/worker-stdout.jsonl';lines=[]
        for line in log.read_text().splitlines():
            outer=json.loads(line)
            try:event=json.loads(outer['line'])
            except json.JSONDecodeError:lines.append(line);continue
            if isinstance(event,dict):
                kind=event.get('event')
                if mutation=='no_combined' and kind=='native_batch':event['active_rows']=[1,0]
                if mutation=='reused_epochs' and kind=='completed':event['native_epoch']=1
                if mutation=='bad_close' and kind=='closed':event['memory']['metal_allocated_bytes']=1024**3
                if mutation=='bad_text' and kind=='completed':event['text']='corrupted'
                if mutation=='cached_output' and kind in ('token','completed'):event['cached_tokens']=1
                outer['line']=json.dumps(event)
            lines.append(json.dumps(outer))
        log.write_text('\n'.join(lines)+'\n');result=audit(path)
        assert result['status']=='failed' and result['issues'],mutation
        print('batch_quality_audit rejects',mutation)
print('batch_quality_audit PASS baseline and five tamper cases')
