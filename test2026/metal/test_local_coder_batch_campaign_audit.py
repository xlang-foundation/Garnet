"""Reject false throughput, prompt, batching and native completion evidence."""
import json,shutil,sys,tempfile
from pathlib import Path
from audit_local_coder_batch_campaign import audit
root=Path(sys.argv[1]).resolve();baseline=audit(root)
expected={'throughput':'Aggregate throughput arithmetic differs','prompt':'Frozen prompt token IDs differ',
    'no_combined':'No actual combined decode','completion':'SSE request has no native completion','epoch':'Native request epochs reused'}
for mutation in ('throughput','prompt','no_combined','completion','epoch'):
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)
        for item in root.iterdir():
            if item.name=='campaign':shutil.copytree(item,path/item.name)
            elif item.name=='server':
                (path/item.name).mkdir();shutil.copy2(item/'worker-stdout.jsonl',path/item.name/'worker-stdout.jsonl')
            else:(path/item.name).symlink_to(item)
        if mutation=='throughput':
            summary=next((path/'campaign/cohort-1').glob('*measured-0/summary.json'))
            data=json.loads(summary.read_text());data['aggregate_output_tokens_per_second']*=2;summary.write_text(json.dumps(data))
        else:
            log=path/'server/worker-stdout.jsonl';lines=[];changed=False
            for line in log.read_text().splitlines():
                outer=json.loads(line)
                try:event=json.loads(outer['line'])
                except json.JSONDecodeError:lines.append(line);continue
                if isinstance(event,dict):
                    kind=event.get('event')
                    if mutation=='no_combined' and kind=='native_batch':event['active_rows']=[1,0]
                    elif mutation=='prompt' and kind=='completed' and not changed:event['prompt_ids'][0]+=1;changed=True
                    elif mutation=='completion' and kind=='completed' and not changed:changed=True;continue
                    elif mutation=='epoch' and kind=='completed':event['native_epoch']=1
                outer['line']=json.dumps(event);lines.append(json.dumps(outer))
            log.write_text('\n'.join(lines)+'\n')
        result=audit(path);assert result['status']=='failed' and expected[mutation] in result['issues'] and expected[mutation] not in baseline['issues'],mutation
        print('batch campaign audit rejects',mutation)
print('batch campaign audit baseline status:',baseline['status'])
print('batch campaign audit PASS five distinct tamper detections; baseline failures retained')
