"""CPU reconstruction of actual MTP work; deduplicate batch-shared counters."""
from pathlib import Path
import hashlib,json,sys,zipfile
source,destination=map(Path,sys.argv[1:3]);pilot='kv-delta-qpc-http-pilot-attempt1'
sha=lambda b:hashlib.sha256(b).hexdigest()
proof_path=source/(pilot+'-verified.json');proof=json.loads(proof_path.read_bytes())
archive=source/(pilot+'-evidence.zip');assert proof['status']=='compact_kv_delta_qpc_http_pilot_replayed'
assert sha(archive.read_bytes())==proof['archive_sha256'];base='results/'+pilot+'/native/';cohorts={}
with zipfile.ZipFile(archive) as z:
    complete={}
    for line in z.read(base+'worker/worker-stdout.jsonl').splitlines():
        try:event=json.loads(json.loads(line)['line'])
        except ValueError:continue
        if isinstance(event,dict) and event.get('event')=='completed':
            assert event['id'] not in complete;complete[event['id']]=event
    dispatch=[json.loads(line) for line in z.read(base+'worker/batch-dispatch.jsonl').splitlines()]
    group_for={key:(index,group) for index,group in enumerate(dispatch) for key in group['request_ids']}
    manifests=[n for n in z.namelist() if n.startswith(base+'benchmark/') and n.endswith('-manifest.json')];assert len(manifests)==1
    manifest=json.loads(z.read(manifests[0]));prefix=manifests[0].rsplit('/',1)[0]+'/'
    for cohort in manifest['cohorts']:
        phase=cohort['case']+'-c'+str(cohort['concurrency']);used={};ids=[]
        for key in cohort['request_keys']:
            finals=[json.loads(event['data']) for line in z.read(prefix+key+'-events.jsonl').splitlines()
                    if (event:=json.loads(line))['data']!='[DONE]' and 'response_token_ids' in json.loads(event['data'])]
            assert len(finals)==1;request_id=finals[0]['id'];index,group=group_for[request_id];used[index]=group;ids.append(request_id)
            assert finals[0]['response_token_ids']==complete[request_id]['generated_ids'] and len(finals[0]['response_token_ids'])==128
        assert {key for g in used.values() for key in g['request_ids']}==set(ids)
        rows=[]
        for index,group in sorted(used.items()):
            first=complete[group['request_ids'][0]];mode=first['generation_mode']
            assert all(complete[key]['generation_mode']==mode for key in group['request_ids'])
            if mode=='greedy_target':continue
            assert mode=='verified_mtp' and group['real_rows']==4
            metrics=first['speculative_metrics']
            assert all(complete[key]['speculative_metrics']==metrics for key in group['request_ids'])
            iterations=metrics['iterations'];assert len(iterations)==metrics['target_verification_calls']>0
            assert metrics['target_forward_calls']==metrics['target_verification_calls']+metrics['target_prefix_recomputes']
            counts=[it['processed_target_counts'] for it in iterations]
            assert all(len(c)==4 and all(type(n) is int and 0<=n<=4 for n in c) for c in counts)
            processed=sum(map(sum,counts));proposed=metrics['proposed_draft_tokens'];accepted=metrics['accepted_draft_tokens']
            assert 0<=accepted<=proposed and proposed>0
            rows.append({'dispatch_index':index,'request_ids':group['request_ids'],'batch_shared_metrics':metrics,
                'processed_target_tokens':processed,'average_processed_per_row_iteration':processed/(4*len(iterations)),
                'accepted_draft_fraction':accepted/proposed,
                'scope':'Counters shared by four completed row events are counted once. Processed target counts include per-row accepted prefixes, not output token arrivals or GPU durations.'})
        assert sum(len(row['request_ids']) for row in rows)==proof['actual_dispatch_by_cohort'][phase]['mtp_requests']
        cohorts[phase]={'measured_requests':8,'mtp_groups':rows,'greedy_requests':proof['actual_dispatch_by_cohort'][phase]['greedy_requests']}
result={'status':'qpc_http_mtp_work_reconstructed','archive_sha256':proof['archive_sha256'],'proof_sha256':sha(proof_path.read_bytes()),
    'analyzer_sha256':sha(Path(__file__).read_bytes()),'cohorts':cohorts,'two_x_established':False,
    'scope':'All8cohorts retained, including greedy-only cohorts. Batch-shared MTP counters are deduplicated and replayed from raw completed events. Does not infer CUDA time, bandwidth, benefit of a different depth or a speedup.'}
destination.write_text(json.dumps(result,indent=2),encoding='utf8')
for key,cohort in cohorts.items():
    print(json.dumps({'cohort':key,'greedy_requests':cohort['greedy_requests'],'mtp_groups':[{n:g[n] for n in ('dispatch_index','average_processed_per_row_iteration','accepted_draft_fraction')} for g in cohort['mtp_groups']]}))
