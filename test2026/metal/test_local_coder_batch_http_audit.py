"""Tamper an archived finite study to verify that its audit rejects corruption.

Argument: completed study directory. CPU-only; never benchmark evidence.
"""
import json,shutil,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_http_study import audit
root=Path(sys.argv[1]);assert audit(root)['status']=='passed'

def mutate_native(path,kind):
    report=json.loads((path/'result.json').read_text());labels={item['label']:item for item in report['responses']}
    rawpath=path/'server/worker-stdout.jsonl';lines=[];a_epoch=None
    for line in rawpath.read_text().splitlines():
        outer=json.loads(line)
        try:event=json.loads(outer['line'])
        except json.JSONDecodeError:lines.append(line);continue
        if isinstance(event,dict):
            if event.get('event')=='completed' and event.get('id')==labels['A']['http_id']:a_epoch=event['native_epoch']
            if kind=='no_combined' and event.get('event')=='native_batch':event['active_rows']=[1,0]
            if kind=='reused_epoch' and event.get('event')=='completed' and event.get('id')==labels['C']['http_id']:event['native_epoch']=a_epoch
            if kind=='bad_close' and event.get('event')=='closed':event['memory']['metal_allocated_bytes']=1024**3
            if kind=='bad_tokens' and event.get('event')=='completed' and event.get('id')==labels['A']['http_id']:event['generated_ids'][0]+=1
            outer['line']=json.dumps(event)
        lines.append(json.dumps(outer))
    rawpath.write_text('\n'.join(lines)+'\n')

for kind in ('no_combined','reused_epoch','bad_close','bad_tokens','bad_cache'):
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'study';shutil.copytree(root,path)
        if kind=='bad_cache':
            report=json.loads((path/'result.json').read_text())
            for row in report['responses']:
                if row['label']=='C-returned':row['cached_tokens']=0;(path/'C-returned/result.json').write_text(json.dumps(row))
            (path/'result.json').write_text(json.dumps(report))
        else:mutate_native(path,kind)
        result=audit(path);assert result['status']=='failed' and result['issues'],kind
        print('batch_http_audit rejects',kind)
print('batch_http_audit PASS baseline and five tamper cases')
