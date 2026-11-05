# SPDX-License-Identifier: Apache-2.0
"""Audit native cached-progress HTTP against frozen and serial fresh controls.

No comparator speed, arbitrary prefix truncation, full4K, peak-memory or coding
quality claim. Native metadata/counters and HTTP outputs are saved observations.
"""
import argparse,copy,hashlib,json
from pathlib import Path
from audit_mistral_http_admission import load_client,require,validate
from audit_mistral_continuous_http import validate_lifecycle


def cached_lifecycle(events,clients):
    by={c['label']:c for c in clients}
    report=validate_lifecycle(events,[by[k] for k in ('A','B','C','D','E','A-return')])
    completed={e['id']:(i,e) for i,e in enumerate(events) if e['event']=='completed'}
    observations={label:completed[c['id']] for label,c in by.items()}
    row=observations['A'][1]['native_row']
    previous=observations['A-return'][0]
    epochs=[observations[k][1]['native_epoch'] for k in ('A','E','A-return')]
    for label in ('append','changed'):
        index,event=observations[label]
        start=next(i for i,e in enumerate(events) if e['event']=='native_batch' and e['stage']=='prefill'
                   and any(r['id']==event['native_request'] for r in e['requests']))
        require(previous<start<index<min(observations[k][0] for k in ('B','C','D')), 'edited return missed survivor overlap')
        require(event['native_row']==row and event['native_epoch']>epochs[-1], 'edited return lost row/epoch')
        epochs.append(event['native_epoch']);previous=index
    expected={'A-return':84,'append':64,'changed':64}
    for label,(_,event) in observations.items():
        require(event['cached_tokens']==expected.get(label,0), 'unexpected restored prefix length')
    stats=[e['stats'] for e in events if e['event']=='native_prefix_stats']
    require(stats and stats[-1]['hits']==3 and stats[-1]['misses']==5, 'native cache hit/miss coverage differs')
    previous_hits=previous_misses=0
    for s in stats:
        require(0<=s['entries']<=16 and 0<=s['accounted_host_bytes']<=512*1024*1024
                and s['hits']>=previous_hits and s['misses']>=previous_misses, 'cache budget/counters violated')
        previous_hits,previous_misses=s['hits'],s['misses']
    report.update(epochs=epochs,cached_tokens={label:e['cached_tokens'] for label,(_,e) in observations.items()},
                  maximum_accounted_host_bytes=max(s['accounted_host_bytes'] for s in stats))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--coding-control',type=Path,required=True)
    p.add_argument('--control-provenance',type=Path,required=True)
    p.add_argument('--control-provenance-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();files={}
    def raw(path):
        data=path.read_bytes();files[str(path.resolve())]=hashlib.sha256(data).hexdigest();return data
    def read(path):return json.loads(raw(path))
    provenance=read(a.control_provenance)
    require(files[str(a.control_provenance.resolve())]==a.control_provenance_sha256,'changed control provenance')
    root=a.study;manifest,status=read(root/'manifest.json'),read(root/'status.json')
    fixture_path=Path(manifest['fixture']);fixture=read(fixture_path)
    require(provenance['files'].get(str(fixture_path.resolve()))==files[str(fixture_path.resolve())],'unfrozen fixture')
    coding=[c for c in fixture['cases'] if c['kind']=='coding'];controls={}
    for c in coding:
        path=a.coding_control/c['id']/'request-0.result.json';controls[c['id']]=read(path)
        require(provenance['files'].get(str(path.resolve()))==files[str(path.resolve())],'unfrozen control')
    require(status['status']=='collected_pending_independent_audit','collection failed')
    entries=manifest['requests'];require([e['label'] for e in entries]==['A','B','C','D','E','A-return','append','changed'],'changed sequence')
    require([e['body']['max_tokens'] for e in entries]==[8,256,320,384,16,32,16,16],'changed budgets')
    base=entries[0]['body']['messages'][0]['content']
    require(entries[6]['body']['messages'][0]['content']==base+'\nInclude a short docstring.'
            and entries[7]['body']['messages'][0]['content']==base.replace('imports needed.','imports required.'),'changed edit contract')
    reports={};parsed={};native={}
    for phase,policy in [('fresh',False),('cached',True)]:
        state=status['phases'][phase]
        require(state['status']=='collected' and state['shutdown']['graceful_native_closed'] is True
                and state['shutdown']['forced_kill'] is False,'unclean phase shutdown')
        folder=root/phase;profile=read(folder/'profile.json')
        events=[json.loads(json.loads(line)['line']) for line in raw(folder/'server/worker-stdout.jsonl').splitlines()]
        native[phase]=events;clients=[]
        selected=entries[6:] if phase=='fresh' else entries
        for i,entry in enumerate(selected):
            label=entry['label'];prefix=folder/label/'request-0'
            if label in ('append','changed') and phase=='fresh':
                wire=[json.loads(line) for line in raw(prefix.with_suffix('.events.jsonl')).splitlines()]
                final=next(json.loads(w['data']) for w in reversed(wire) if w['data']!='[DONE]' and 'prompt_token_ids' in json.loads(w['data']))
                prompt=final['prompt_token_ids']
                require(prompt[:64]==coding[0]['prompt_ids'][:64] and prompt!=coding[0]['prompt_ids'],'edit lost stored checkpoint')
            elif label in ('append','changed'):prompt=controls['new-'+label]['prompt_ids']
            else:prompt=coding[0 if label=='A-return' else i]['prompt_ids']
            client=load_client(prefix,entry['case_id'],entry['body'],prompt,read,raw);client['label']=label;clients.append(client)
            if phase=='fresh':
                controls[entry['case_id']]=dict(status='completed',prompt_ids=prompt,generated_ids=client['sse_ids'],text=client['result']['text'])
        parsed[phase]=clients
        reports[phase]=validate(events,clients,controls,profile,read(folder/'server/worker-exit.json'),
            expected_requests=len(clients),control_prefixes=True,maximum_chunk=64,prefix_policy=policy,required_active_rows=4 if policy else 1)
    reports['lifecycle']=cached_lifecycle(native['cached'],parsed['cached'])
    require(reports['cached']['head_only_prefill_batches']>=1,'exact return ran prefill body')
    rejected=[]
    for label in ('false-hit','no-survivor-overlap','reused-epoch','over-budget'):
        altered=copy.deepcopy(native['cached']);by={c['label']:next(e for e in altered if e['event']=='completed' and e['id']==c['id']) for c in parsed['cached']}
        if label=='false-hit':by['A-return']['cached_tokens']=0
        elif label=='no-survivor-overlap':
            event=by['B'];altered.remove(event);altered.insert(altered.index(by['E']),event)
        elif label=='reused-epoch':by['changed']['native_epoch']=by['append']['native_epoch']
        else:next(e for e in altered if e['event']=='native_prefix_stats')['stats']['accounted_host_bytes']=513*1024*1024
        try:cached_lifecycle(altered,parsed['cached'])
        except (ValueError,KeyError,StopIteration):rejected.append(label)
        else:raise AssertionError('Accepted cached lifecycle corruption: '+label)
    cached_profile=read(root/'cached/profile.json');cached_exit=read(root/'cached/server/worker-exit.json')
    for label in ('forged-head-offset','missing-prefill-body','extra-head-body','unstable-sse-cache'):
        altered=copy.deepcopy(native['cached']);clients=copy.deepcopy(parsed['cached'])
        returned=next(e for e in altered if e['event']=='completed' and e['id']==next(c['id'] for c in clients if c['label']=='A-return'))
        head=next(e for e in altered if e['event']=='native_batch' and e['stage']=='prefill'
                  and any(r['id']==returned['native_request'] for r in e['requests']))
        if label=='forged-head-offset':head['requests'][0]['prefill_offset']-=1
        elif label=='missing-prefill-body':altered.remove(next(e for e in altered if e['event']=='native_affine_profile' and e['stage']=='prefill'))
        elif label=='extra-head-body':
            body=copy.deepcopy(next(e for e in altered if e['event']=='native_affine_profile' and e['stage']=='prefill'))
            altered.insert(altered.index(head)+1,body)
        else:next(c for c in clients if c['label']=='A-return')['stream_cached_counts']=[0,84]
        try:
            validate(altered,clients,controls,cached_profile,cached_exit,expected_requests=8,control_prefixes=True,
                     maximum_chunk=64,prefix_policy=True,required_active_rows=4)
        except (ValueError,KeyError,StopIteration):rejected.append(label)
        else:raise AssertionError('Accepted cached core corruption: '+label)
    for name in ('audit_mistral_cached_http.py','audit_mistral_http_admission.py','audit_mistral_continuous_http.py'):
        raw(Path(__file__).with_name(name))
    report=dict(cached_http_gate_passed=True,phases=reports,corruptions_rejected=rejected,
                output_token_observations=sum(len(c['sse_ids']) for group in parsed.values() for c in group),scope=__doc__,files=files)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,observations=report['output_token_observations'],head_only_batches=reports['cached']['head_only_prefill_batches'],
                         cached_tokens=reports['lifecycle']['cached_tokens'],maximum_host_bytes=reports['lifecycle']['maximum_accounted_host_bytes'])))


if __name__=='__main__':main()
