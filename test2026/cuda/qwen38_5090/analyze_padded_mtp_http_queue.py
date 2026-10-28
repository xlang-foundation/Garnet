"""Replay all measured queue traces using QPC stamps from one bridge process."""
from pathlib import Path
import hashlib,json,statistics,sys,zipfile
source,destination=map(Path,sys.argv[1:3]);name='padded-mtp-http-pilot-attempt1'
sha=lambda b:hashlib.sha256(b).hexdigest()
proof_path=source/(name+'-verified.json');proof=json.loads(proof_path.read_bytes())
assert proof['status']=='padded_mtp_http_pilot_replayed' and proof['grouping_clock']=='perf_counter' and proof['measured_sse_worker_pairs']==64
archive=source/(name+'-evidence.zip');assert sha(archive.read_bytes())==proof['archive_sha256']
base='results/'+name+'/native/';cohorts={}
with zipfile.ZipFile(archive) as z:
    paths=[p for p in z.namelist() if p.startswith(base+'benchmark/') and p.endswith('-manifest.json')];assert len(paths)==1
    manifest=json.loads(z.read(paths[0]));prefix=paths[0].rsplit('/',1)[0]+'/'
    groups=[json.loads(line) for line in z.read(base+'worker/batch-dispatch.jsonl').splitlines()]
    mapping={key:(n,g) for n,g in enumerate(groups) for key in g['request_ids']}
    assert len(mapping)==sum(len(g['request_ids']) for g in groups)
    for cohort in manifest['cohorts']:
        phase=cohort['case']+'-c'+str(cohort['concurrency']);requests=[];used={}
        for key in cohort['request_keys']:
            sse=[json.loads(line) for line in z.read(prefix+key+'-events.jsonl').splitlines()]
            finals=[json.loads(event['data']) for event in sse if event['data']!='[DONE]' and 'response_token_ids' in json.loads(event['data'])]
            assert len(finals)==1;request_id=finals[0]['id'];number,group=mapping[request_id];used[number]=group
            created=group['request_created_perf_counter'][request_id];enqueued=group['queue_put_started_perf_counter'][request_id]
            dispatched=group['dispatch_perf_counter'];start=group['grouping_started_perf_counter'];deadline=group['grouping_deadline_perf_counter']
            assert created<=enqueued<=dispatched and start<=dispatched and abs(deadline-start-.010)<1e-8
            if group['real_rows']<4:assert dispatched>=deadline
            requests.append({'key':key,'request_id':request_id,'group_number':number,'real_rows':group['real_rows'],
                'created_perf_counter':created,'queue_put_started_perf_counter':enqueued,'dispatch_perf_counter':dispatched,
                'queue_put_to_dispatch_ms':(dispatched-enqueued)*1000,
                'created_to_queue_put_ms':(enqueued-created)*1000})
        ordered=sorted(used);assert len(requests)==8 and len(ordered)==proof['actual_dispatch_by_cohort'][phase]['groups']
        origin=used[ordered[0]]['grouping_started_perf_counter']
        for request in requests:request['handler_generation_entry_relative_ms']=(request['created_perf_counter']-origin)*1000
        cohorts[phase]={'requests':requests,'groups':[{'number':n,'real_rows':used[n]['real_rows'],
            'grouping_elapsed_ms':(used[n]['dispatch_perf_counter']-used[n]['grouping_started_perf_counter'])*1000,
            'started_relative_ms':(used[n]['grouping_started_perf_counter']-origin)*1000,
            'deadline_relative_ms':(used[n]['grouping_deadline_perf_counter']-origin)*1000} for n in ordered],
            'queue_put_to_dispatch_median_ms':statistics.median(r['queue_put_to_dispatch_ms'] for r in requests),
            'handler_entry_spread_ms':(max(r['created_perf_counter'] for r in requests)-min(r['created_perf_counter'] for r in requests))*1000}
assert set(cohorts)=={case+'-c'+str(c) for case in ('text','image') for c in (1,2,4,8)}
result={'status':'closed_qpc_http_queue_reconstructed','archive_sha256':proof['archive_sha256'],'proof_sha256':sha(proof_path.read_bytes()),
    'analyzer_sha256':sha(Path(__file__).read_bytes()),'cohorts':cohorts,'measured_requests':64,'two_x_established':False,
    'scope':'All64measured requests retained. Created/queue-put/grouping/dispatch stamps use QPC in the same bridge process. Queue-put stamp precedes Queue.put and dispatch stamp precedes JSON log/pipe write. These are host observations, not GPU timings or a causal speed comparison. HTTP accept and complete-body-read timestamps remain absent; pre-enqueue delay cannot be assigned to sockets or handler scheduling.'}
destination.write_text(json.dumps(result,indent=2),encoding='utf8')
for phase,cohort in cohorts.items():print(json.dumps({'cohort':phase,'real_rows':[g['real_rows'] for g in cohort['groups']],
    'queue_put_to_dispatch_median_ms':cohort['queue_put_to_dispatch_median_ms'],'handler_entry_spread_ms':cohort['handler_entry_spread_ms']}))
