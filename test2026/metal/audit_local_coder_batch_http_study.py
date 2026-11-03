"""Audit the finite native HTTP refill/prefix study after worker shutdown."""
import argparse,hashlib,json
from pathlib import Path

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def audit(root):
    issues=[];manifest=json.loads((root/'manifest.json').read_text());report=json.loads((root/'result.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(root/'frozen'/name)!=expected:issues.append('Frozen source/asset hash mismatch: '+name)
    if (root/'libgarnet.dylib').exists() and digest(root/'libgarnet.dylib')!=manifest['library_sha256']:issues.append('Native library hash mismatch')
    if report['status']!='collected_pending_native_audit' or report['issues']:issues.append('Collection did not finish cleanly')
    cases=json.loads((root/'frozen/cases.json').read_text());ready=[];closed=[];native_errors=[];streams={};completed={};batches=[]
    for number,line in enumerate((root/'server/worker-stdout.jsonl').read_text().splitlines()):
        try:event=json.loads(json.loads(line)['line'])
        except json.JSONDecodeError:continue
        if not isinstance(event,dict):continue
        kind=event.get('event')
        if kind=='ready':ready.append(event)
        elif kind=='closed':closed.append(event)
        elif kind=='native_batch':batches.append((number,event))
        elif kind=='error':native_errors.append(event)
        elif kind=='token':
            key=event['id'];tokens=streams.setdefault(key,[]);tokens.append(event['token_id'])
            if event['sampled_tokens']!=len(tokens):issues.append('Native token usage is not sequential')
        elif kind=='completed':
            key=event['id']
            if key in completed:issues.append('Duplicate native completion identity')
            completed[key]=(number,event)
            if streams.get(key)!=event['generated_ids']:issues.append('Native streamed/final token mismatch')
    if len(ready)!=1 or ready[0].get('active_requests')!=2 or ready[0].get('prefix_cache') is not True:issues.append('Ready profile differs')
    if len(closed)!=1 or closed[0]['memory']['metal_allocated_bytes']!=720896:issues.append('Native close allocation differs')
    exit_record=json.loads((root/'server/worker-exit.json').read_text())
    if exit_record['returncode']!=0 or not exit_record['expected_close']:issues.append('Native worker did not exit cleanly')
    if len(native_errors)!=1 or native_errors[0].get('fatal') is not False or 'exceeds' not in native_errors[0].get('error',''):issues.append('Unexpected native rejection/failure')
    labels=['A','B','C','C-returned'];by_label={row['label']:row for row in report['responses']}
    if sorted(by_label)!=sorted(labels) or len(completed)!=4 or set(completed)!=set(by_label[label]['http_id'] for label in labels):issues.append('Native/client completion set differs')
    rows={}
    for label,case in zip(labels,cases+[cases[2]]):
        key=by_label[label]['http_id'];index,event=completed[key];rows[label]=(index,event)
        local=json.loads((root/label/'result.json').read_text())
        if local!=by_label[label]:issues.append(label+': stored client result changed')
        if event['prompt_ids']!=case['prompt_ids'] or event['generated_ids']!=case['expected_ids']:issues.append(label+': frozen token oracle differs')
        if local['generated_ids']!=event['generated_ids'] or local['cached_tokens']!=event['cached_tokens']:issues.append(label+': native/client usage differs')
        expected_cache=len(case['prompt_ids']) if label=='C-returned' else 0
        if event['cached_tokens']!=expected_cache:issues.append(label+': prefix reuse differs')
        ids=[];last=None;done=0
        for raw in (root/label/'events.jsonl').read_text().splitlines():
            payload=json.loads(raw)['data']
            if payload=='[DONE]':done+=1;continue
            item=json.loads(payload)
            if item.get('id')!=key:issues.append(label+': SSE identity mismatch')
            if 'token_id' in item:ids.append(item['token_id'])
            if item['usage']['prompt_tokens_details']['cached_tokens']!=expected_cache:issues.append(label+': SSE cached-token mismatch')
            last=item
        if done!=1 or ids!=event['generated_ids'] or last['response_token_ids']!=ids or last['prompt_token_ids']!=case['prompt_ids']:issues.append(label+': SSE/final IDs mismatch')
        if last['usage']['completion_tokens']!=len(ids) or last['usage']['prompt_tokens']!=len(case['prompt_ids']):issues.append(label+': SSE usage differs')
        if event['memory']['metal_allocated_bytes']>=event['memory']['metal_recommended_working_set_bytes']:issues.append(label+': device working set exceeded')
    if not any(event['stage']=='decode' and sum(event['active_rows'])==2 for _,event in batches):issues.append('No combined native decode plan')
    ai,a=rows['A'];bi,b=rows['B'];ci,c=rows['C']
    if a['native_row']!=c['native_row'] or a['native_epoch']==c['native_epoch']:issues.append('A-to-C row replacement/epoch differs')
    prefill=[index for index,event in batches if event['stage']=='prefill' and any(request['id']==c['native_request'] for request in event['requests'])]
    if len(prefill)!=1 or not ai<prefill[0]<bi:issues.append('C was not admitted while B continued after A finished')
    return dict(status='passed' if not issues else 'failed',issues=issues,
        scope='Finite learned HTTP token routing, combined decode, refill, prefix counts and shutdown; not full coding quality or performance qualification',
        completions=len(completed),combined_decode_plans=sum(event['stage']=='decode' and sum(event['active_rows'])==2 for _,event in batches),
        returned_cached_tokens=rows['C-returned'][1]['cached_tokens'],closed_metal_bytes=closed[0]['memory']['metal_allocated_bytes'] if closed else None,
        manifest_sha256=digest(root/'manifest.json'),worker_log_sha256=digest(root/'server/worker-stdout.jsonl'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=audit(args.study)
    with args.output.open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if report['issues']:raise SystemExit(1)
if __name__=='__main__':main()
