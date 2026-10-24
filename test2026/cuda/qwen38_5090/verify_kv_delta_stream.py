"""Replay compact protocol bytes/tokens against qualified retained stream."""
from pathlib import Path
import base64,hashlib,json,math,sys,zipfile
out=Path(sys.argv[1]);name='kv-delta-stream-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
def protocol(raw):
    events=[]
    for line in raw.decode('utf8').splitlines():
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        if event.get('event') in ('ready','token','completed','error'):events.append(event)
    return events
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';w=doc(base+'workflow.json');passed=w['exit_code']==0
    assert w['cleanup_finished'] and w['library_unchanged'] and w['library_sha256']==m['library_sha256']
    assert w['source_sha256']==w['source_after_sha256'] and w['stream_source_sha256']==w['stream_source_after_sha256']
    methods='Garnet/test2026/cuda/qwen38_5090/';prepraw=z.read(methods+'evidence/kv-delta-stream-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==w['preparation_sha256'] and prep['groups']==13 and prep['requests']==40
    assert sha(z.read(methods+'run_kv_delta_stream.py'))==prep['runner_sha256']
    for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h==w['stream_source_sha256'][n]
    for source,target in (('serve_native_kv_delta_worker.bat','serve-native-kv-delta-worker.bat'),('test_kv_delta_stream.bat','test-kv-delta-stream.bat')):assert z.read(methods+source)==z.read(target)
    full='kv-delta-full-b4-t4-attempt1';fullproof=doc('results/'+full+'-verified.json');fullw=doc('results/'+full+'/workflow.json')
    assert fullproof['status']=='compact_kv_delta_full_model_replayed' and fullproof['raw_last_hidden_all_exact'] and fullproof['raw_last_hidden_arrays_compared']==8
    assert len(fullproof['cases'])==4 and all(c['same_DLL_default_off_exact'] and c['same_profile_state_exact'] and c['ids_exact'] for c in fullproof['cases'])
    assert fullproof['library_sha256']==m['library_sha256'] and fullproof['archive_sha256']==w['full_model_archive_sha256']==m['linked_archives'][full+'-evidence.zip']
    assert sha(z.read(methods+'verify_kv_delta_full_b4_t4.py'))==fullproof['replay_verifier_sha256']
    assert w['source_sha256']=={**fullw['candidate_source_sha256'],**fullw['unchanged_parent_source_sha256'],**fullw['unchanged_native_source_sha256']}
    with zipfile.ZipFile(out/(full+'-evidence.zip')) as model:
        assert z.read('results/'+full+'/workflow.json')==model.read('results/'+full+'/workflow.json')
    repair=doc('results/unicode-transport-repair2-verified.json');assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
    assert sha(z.read(methods+'native_worker_io.py'))==repair['repair_source_sha256']
    old='b4-host-joined-mtp-stream-attempt1';proof=doc('results/'+old+'-verified.json')
    assert proof['status']=='b4_host_joined_mtp_stream_replayed' and proof['groups']==13 and proof['requests']==40 and proof['exact_ids_and_prompt_ids'] and proof['image_hashes_verified']
    assert proof['archive_sha256']==w['retained_stream_archive_sha256']==prep['retained_stream_archive_sha256']==m['linked_archives'][old+'-evidence.zip']
    peak=None;requests=0;wire_sha=None;modes={}
    with zipfile.ZipFile(out/(old+'-evidence.zip')) as previous:
        oldbase='results/'+old+'/native/'
        for key,source,target in (('worker_derivation','native_b4_host_joined_mtp_stream_worker.py','native_kv_delta_stream_worker.py'),('probe_derivation','probe_b4_host_joined_mtp_stream.py','probe_kv_delta_stream.py')):
            text=previous.read(methods+source).decode().replace('\r\n','\n')
            for before,after,count in prep[key]:assert text.count(before)==count;text=text.replace(before,after)
            assert text.encode()==z.read(methods+target)
        if passed:
            report=doc(base+'native/report.json');assert report['exit_code']==0 and report['status']=='batch_stream_protocol_passed_needs_independent_replay'
            wire=z.read(base+'native/requests.jsonl');wire_sha=sha(wire);assert wire==previous.read(oldbase+'requests.jsonl') and wire_sha==report['request_sha256']
            assert z.read(base+'native/teacher-provenance.json')==previous.read(oldbase+'teacher-provenance.json')
            frames=[json.loads(l) for l in wire.splitlines()];assert len(frames)==13
            reference=protocol(previous.read(oldbase+'worker-stdout.log'));old_completed={e['id']:e for e in reference if e['event']=='completed'}
            old_tokens={key:[e for e in reference if e['event']=='token' and e['id']==key] for key in old_completed}
            assert len(old_completed)==40;expected={}
            for frame in frames:
                jobs=frame['batch'];use_mtp=len(jobs)==4 and len({j['body']['max_tokens'] for j in jobs})==1
                for job in jobs:
                    key=job['id'];ref=old_completed[key];content=job['body']['messages'][0]['content']
                    image=None if isinstance(content,str) else sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True))
                    assert image==ref['image_sha256']
                    expected[key]={'reference':ref,'mode':'verified_mtp' if use_mtp else 'greedy_target','image':image,'limit':job['body']['max_tokens']}
            tokens={k:[] for k in expected};done=set();ready=[];last_time=-math.inf
            for event in protocol(z.read(base+'native/worker-stdout.log')):
                kind=event['event']
                if kind=='ready':ready.append(event);continue
                assert len(ready)==1 and kind!='error';key=event['id'];assert key in expected and key not in done;ref=expected[key]['reference']
                if kind=='token':
                    index=len(tokens[key]);old_event=old_tokens[key][index]
                    assert all(event[k]==old_event[k] for k in ('token_id','sampled_tokens','prompt_tokens','text','stop'))
                    tokens[key].append(event['token_id']);now=event['native_monotonic_seconds'];assert math.isfinite(now) and now>=last_time;last_time=now
                    assert event['sampled_tokens']==len(tokens[key]) and event['prompt_tokens']==len(ref['prompt_ids'])
                else:
                    assert event['generated_ids']==tokens[key]==ref['generated_ids'] and event['prompt_ids']==ref['prompt_ids']
                    assert event['generation_mode']==expected[key]['mode']==ref['generation_mode'] and event['image_sha256']==expected[key]['image']
                    assert event['text']==ref['text'] and event['finish_reason']==ref['finish_reason'] and 1<=len(tokens[key])<=expected[key]['limit']
                    if event['generation_mode']=='verified_mtp':
                        metrics=event['speculative_metrics'];assert metrics['target_forward_calls']==metrics['target_verification_calls']+metrics['target_prefix_recomputes']
                        assert metrics['target_verification_calls']==len(metrics['iterations']) and 0<=metrics['accepted_draft_tokens']<=metrics['proposed_draft_tokens'] and metrics['mtp_prefill_calls']>0
                    modes[key]=event['generation_mode'];done.add(key)
            assert done==set(expected) and len(done)==40 and len(ready)==1 and ready[0]==report['ready']
            r=ready[0];assert r['protocol']=='study_batch_v1' and r['batch']==4 and r['capacity']==4096 and r['prefill_chunk']==r['chunk']==4 and r['mtp_depth']==3
            assert r['kv_delta_state'] and r['shared_output_storage'] and r['deferred_state_commit'] and r['host_embedding_lookup'] and r['joined_mlp_projections'] and r['shared_compressed_weights']
            assert r['embedding_mode']=='host_mapped_bf16' and r['embedding_table_bytes']==0 and r['mtp_policy']=='four_real_rows_equal_limits_else_greedy' and r['library_sha256']==m['library_sha256']
            assert {(c['id'],c['tokens'],c['exact']) for c in report['cases']}=={(k,len(v['reference']['generated_ids']),True) for k,v in expected.items()}
            requests=len(done);monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==0 and monitor['sampled_memory']['samples']==len(samples);peak=max(s['used_bytes'] for s in samples);assert peak==monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
            assert monitor['gpu']==json.loads(previous.read('results/'+old+'/resources/resource-monitor.json'))['gpu']
    events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
    if passed:assert w['status']=='compact_stream_finished_needs_independent_replay'
    verifier=sha(z.read(methods+'verify_kv_delta_stream.py'))
result={'status':'compact_kv_delta_stream_replayed' if passed else 'compact_kv_delta_stream_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
        'library_sha256':m['library_sha256'],'groups':13 if passed else 0,'requests':requests,'request_wire_sha256':wire_sha,'generation_mode_by_id':modes,
        'sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':verifier,'performance_admitted':False,
        'scope':'Compact B4/T4 protocol token/prompt/image/text/reset/row-isolation exact against qualified immutable13group/40request stream, with source/full-model/Unicode lineage. Monotonic times validated only as protocol data; no HTTP/actual4K/broadquality/speed/2x admission.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
