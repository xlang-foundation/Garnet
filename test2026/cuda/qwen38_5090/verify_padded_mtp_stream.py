"""Replay private-padding protocol bytes/tokens against qualified retained stream."""
from pathlib import Path
import base64,hashlib,json,math,sys,zipfile
out=Path(sys.argv[1]);name='padded-mtp-stream-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
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
    methods='Garnet/test2026/cuda/qwen38_5090/';prepraw=z.read(methods+'evidence/padded-mtp-stream-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==w['preparation_sha256'] and prep['groups']==13 and prep['requests']==40
    assert sha(z.read(methods+'run_padded_mtp_stream.py'))==prep['runner_sha256']
    for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h==w['stream_source_sha256'][n]
    for source,target in (('serve_native_kv_delta_padded_mtp_worker.bat','serve-native-kv-delta-padded-mtp-worker.bat'),('test_padded_mtp_stream.bat','test-padded-mtp-stream.bat')):assert z.read(methods+source)==z.read(target)
    full='padded-mtp-state-attempt1';fullproof=doc('results/'+full+'-verified.json');fullw=doc('results/'+full+'/workflow.json')
    assert fullproof['status']=='compact_padded_mtp_state_replayed' and fullproof['real_state_all_exact'] and fullproof['raw_ordinary_mtp_hidden_arrays']==24 and fullproof['reset_hidden_arrays']==12
    assert len(fullproof['cases'])==12 and all(c['real_rows_state_exact'] and c['raw_real_hidden_exact'] for c in fullproof['cases']) and fullproof['full_four_default_off_controls']==2
    assert fullproof['library_sha256']==m['library_sha256'] and fullproof['archive_sha256']==w['padded_state_archive_sha256']==m['linked_archives'][full+'-evidence.zip']
    assert sha(z.read(methods+'verify_padded_mtp_state.py'))==fullproof['replay_verifier_sha256']
    assert w['source_sha256']==fullw['source_sha256']
    assert sha(z.read(methods+'native_kv_delta_padded_mtp_stream_worker.py'))==fullproof['candidate_worker_sha256']
    assert sha(z.read(methods+'padded_mtp_policy.py'))==fullproof['padding_policy_sha256']
    config='models/Qwen3.8-27B-NVFP4/config.json'
    assert sha(z.read(config))==w['model_config_sha256']==w['model_config_after_sha256']
    with zipfile.ZipFile(out/(full+'-evidence.zip')) as model:
        assert z.read('results/'+full+'/workflow.json')==model.read('results/'+full+'/workflow.json') and z.read(config)==model.read(config)
    repair=doc('results/unicode-transport-repair2-verified.json');assert repair['status']=='canonical_unicode_transport_repair_independently_verified' and repair['all_repaired_bytes_match_cpython']
    assert sha(z.read(methods+'native_worker_io.py'))==repair['repair_source_sha256']
    old='kv-delta-stream-attempt1';proof=doc('results/'+old+'-verified.json')
    assert proof['status']=='compact_kv_delta_stream_replayed' and proof['groups']==13 and proof['requests']==40 and len(proof['generation_mode_by_id'])==40
    assert proof['archive_sha256']==w['retained_stream_archive_sha256']==prep['retained_stream_archive_sha256']==m['linked_archives'][old+'-evidence.zip']
    peak=None;requests=0;wire_sha=None;modes={}
    with zipfile.ZipFile(out/(old+'-evidence.zip')) as previous:
        oldbase='results/'+old+'/native/'
        baseline=previous.read(methods+'verify_kv_delta_stream.py').decode().replace('\r\n','\n')
        current=Path(__file__).read_text(encoding='utf8')
        for block in prep['verifier_retained_blocks']:
            a=baseline.index(block['begin']);b=baseline.index(block['end'],a);part=baseline[a:b]
            assert sha(part.encode())==block['sha256'] and current.count(part)==1
        for key,source,target in (('worker_derivation','native_kv_delta_stream_worker.py','native_kv_delta_padded_mtp_stream_worker.py'),('probe_derivation','probe_kv_delta_stream.py','probe_padded_mtp_stream.py')):
            text=previous.read(methods+source).decode().replace('\r\n','\n')
            for before,after,count in prep[key]:assert text.count(before)==count;text=text.replace(before,after)
            assert text.encode()==z.read(methods+target)
        if passed:
            report=doc(base+'native/report.json');assert report['exit_code']==0 and report['status']=='batch_stream_protocol_passed_needs_independent_replay'
            wire=z.read(base+'native/requests.jsonl');wire_sha=sha(wire);assert wire==previous.read(oldbase+'requests.jsonl') and wire_sha==report['request_sha256']
            assert z.read(base+'native/teacher-provenance.json')==previous.read(oldbase+'teacher-provenance.json')
            frames=[json.loads(l) for l in wire.splitlines()];assert len(frames)==13
            policy=[{'group':i,'real_rows':len(f['batch']),'generation_mode':'verified_mtp' if len({j['body']['max_tokens'] for j in f['batch']})==1 else 'greedy_target'} for i,f in enumerate(frames)]
            assert policy==prep['expected_policy_by_group'] and wire_sha==prep['request_wire_sha256']
            reference=protocol(previous.read(oldbase+'worker-stdout.log'));old_completed={e['id']:e for e in reference if e['event']=='completed'}
            old_tokens={key:[e for e in reference if e['event']=='token' and e['id']==key] for key in old_completed}
            assert len(old_completed)==40;expected={}
            for frame in frames:
                jobs=frame['batch'];use_mtp=len({j['body']['max_tokens'] for j in jobs})==1
                for job in jobs:
                    key=job['id'];ref=old_completed[key];content=job['body']['messages'][0]['content']
                    image=None if isinstance(content,str) else sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True))
                    assert image==ref['image_sha256']
                    expected[key]={'reference':ref,'mode':'verified_mtp' if use_mtp else 'greedy_target','image':image,'limit':job['body']['max_tokens'],'real_rows':len(jobs)}
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
                    assert event['generation_mode']==expected[key]['mode'] and event['image_sha256']==expected[key]['image']
                    assert event['text']==ref['text'] and event['finish_reason']==ref['finish_reason'] and 1<=len(tokens[key])<=expected[key]['limit']
                    if event['generation_mode']=='verified_mtp':
                        real=expected[key]['real_rows'];assert event['physical_batch']==4 and event['real_rows']==real and event['padded_rows']==4-real
                        assert event['padding_policy']=='duplicate_first_real_row_no_public_dummy_events'
                        if ref['generation_mode']=='verified_mtp':assert event['speculative_metrics']==ref['speculative_metrics']
                        metrics=event['speculative_metrics'];assert metrics['target_forward_calls']==metrics['target_verification_calls']+metrics['target_prefix_recomputes']
                        assert metrics['target_verification_calls']==len(metrics['iterations']) and 0<=metrics['accepted_draft_tokens']<=metrics['proposed_draft_tokens'] and metrics['mtp_prefill_calls']>0
                    modes[key]=event['generation_mode'];done.add(key)
            assert done==set(expected) and len(done)==40 and len(ready)==1 and ready[0]==report['ready']
            assert sum(mode=='verified_mtp' for mode in modes.values())==23 and sum(mode=='greedy_target' for mode in modes.values())==17
            r=ready[0];assert r['protocol']=='study_batch_v1' and r['batch']==4 and r['capacity']==4096 and r['prefill_chunk']==r['chunk']==4 and r['mtp_depth']==3
            assert r['kv_delta_state'] and r['shared_output_storage'] and r['deferred_state_commit'] and r['host_embedding_lookup'] and r['joined_mlp_projections'] and r['shared_compressed_weights']
            assert r['embedding_mode']=='host_mapped_bf16' and r['embedding_table_bytes']==0 and r['mtp_policy']=='equal_real_limits_private_padding_to_four_else_greedy' and r['padding_policy']=='duplicate_first_real_row_no_public_dummy_events' and r['library_sha256']==m['library_sha256']
            assert {(c['id'],c['tokens'],c['exact']) for c in report['cases']}=={(k,len(v['reference']['generated_ids']),True) for k,v in expected.items()}
            requests=len(done);monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==0 and monitor['sampled_memory']['samples']==len(samples);peak=max(s['used_bytes'] for s in samples);assert peak==monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
            assert monitor['gpu']==json.loads(previous.read('results/'+old+'/resources/resource-monitor.json'))['gpu']
    events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
    if passed:assert w['status']=='padded_stream_finished_needs_independent_replay'
    verifier=sha(z.read(methods+'verify_padded_mtp_stream.py'))
result={'status':'compact_padded_mtp_stream_replayed' if passed else 'compact_padded_mtp_stream_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
        'library_sha256':m['library_sha256'],'groups':13 if passed else 0,'requests':requests,'request_wire_sha256':wire_sha,'generation_mode_by_id':modes,
        'sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':verifier,'performance_admitted':False,
        'scope':prep['scope']+' Source/state/Unicode lineage and real-only callback accounting replayed. Physical MTP counters include padding; times are protocol observations only.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
