"""Replay compact HTTP pilot from raw SSE/worker/dispatch and matched archives."""
from pathlib import Path
import base64,hashlib,json,sys,zipfile
out=Path(sys.argv[1]);repo=Path(sys.argv[2]) if len(sys.argv)>2 else Path('D:/CantorAI/work/qwen38-5090')
sys.path.insert(0,str(repo/'test2026/cuda/qwen38_5090'))
from compare_http_runs import verify_run,compare
sha=lambda b:hashlib.sha256(b).hexdigest()
pilot='kv-delta-qpc-http-pilot-attempt1';archive=out/(pilot+'-evidence.zip')
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));evidence=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for n,h in evidence['files'].items():assert sha(z.read(n))==h
    for n,h in evidence['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+pilot+'/';native=base+'native/';workflow=doc(base+'workflow.json')
    assert workflow['cleanup_finished'] and workflow['library_unchanged'] and workflow['library_sha256']==evidence['library_sha256']
    for key,after in (('source_sha256','source_after_sha256'),('stream_source_sha256','stream_source_after_sha256'),('http_helper_sha256','http_helper_after_sha256')):
        assert workflow[key]==workflow[after]
        for n,h in workflow[key].items():
            path='Garnet/'+n if key=='source_sha256' else 'Garnet/test2026/cuda/qwen38_5090/'+n
            assert sha(z.read(path))==h
    methods='Garnet/test2026/cuda/qwen38_5090/';prepraw=z.read(methods+'evidence/kv-delta-qpc-http-preparation.json');prep=json.loads(prepraw)
    assert sha(prepraw)==workflow['preparation_sha256'] and sha(z.read(methods+'run_kv_delta_qpc_http.py'))==prep['runner_sha256']
    assert prep['grouping_delay_ms']==workflow['grouping_delay_ms']==10 and prep['physical_batch']==4 and prep['concurrency']==[1,2,4,8] and prep['requests_per_cohort']==8 and prep['warmups']==2 and prep['max_tokens']==128 and prep['seed']==3
    for n,h in prep['files'].items():assert sha(z.read(methods+n))==h
    for helper in ('compare_http_runs.py','metrics.py','benchmark_stream.py','probe_reference_http.py','sample_vram.py','run_logged.py'):
        assert sha((repo/'test2026/cuda/qwen38_5090'/helper).read_bytes())==sha(z.read(methods+helper))
    cpu=doc(base+'qpc-bridge-cpu-tests.json')
    assert cpu['exit_code']==0 and 'Ran 21 tests' in cpu['stderr'] and cpu['source_sha256']==prep['cpu_sources']
    for n,h in prep['cpu_sources'].items():assert sha(z.read(methods+n))==h
    for n,h in prep['cpu_results'].items():assert sha(z.read('results/'+n))==h
    phase=doc('results/batch-group-clock-phase-cpu-attempt1.json')
    assert phase['status']=='cpu_four_row_group_clock_phase_observed' and len(phase['raw_trials'])==1024
    assert phase['source_sha256']==sha(z.read(methods+'probe_batch_group_clock_phase.py'))
    stream='kv-delta-stream-attempt1';protocol=doc('results/'+stream+'-verified.json')
    assert protocol['status']=='compact_kv_delta_stream_replayed' and protocol['requests']==40 and protocol['groups']==13 and protocol['library_sha256']==evidence['library_sha256']
    assert protocol['archive_sha256']==workflow['stream_archive_sha256']==evidence['linked_archives'][stream+'-evidence.zip']
    streamw=doc('results/'+stream+'/workflow.json');assert workflow['source_sha256']==streamw['source_sha256'] and workflow['stream_source_sha256']==streamw['stream_source_sha256']
    full='kv-delta-full-b4-t4-attempt1';fullproof=doc('results/'+full+'-verified.json')
    assert fullproof['status']=='compact_kv_delta_full_model_replayed' and fullproof['raw_last_hidden_all_exact'] and len(fullproof['cases'])==4 and fullproof['raw_last_hidden_arrays_compared']==8
    assert all(c['same_DLL_default_off_exact'] and c['same_profile_state_exact'] and c['ids_exact'] for c in fullproof['cases']) and fullproof['library_sha256']==evidence['library_sha256']
    assert fullproof['archive_sha256']==workflow['full_model_archive_sha256']==evidence['linked_archives'][full+'-evidence.zip']
    parent_name='kv-delta-http-pilot-attempt1';parent=doc('results/'+parent_name+'-verified.json')
    assert parent['status']=='compact_kv_delta_http_pilot_replayed' and parent['grouping_delay_ms']==10 and parent['archive_sha256']==workflow['parent_http_archive_sha256']==prep['retained_http_archive_sha256']==evidence['linked_archives'][parent_name+'-evidence.zip']
    with zipfile.ZipFile(out/(parent_name+'-evidence.zip')) as previous:
        text=previous.read(methods+'probe_kv_delta_http.py').decode().replace('\r\n','\n')
        for before,after,count in prep['probe_derivation']:assert text.count(before)==count;text=text.replace(before,after)
        assert text.encode()==z.read(methods+'probe_kv_delta_qpc_http.py') and sha(text.encode())==prep['probe_sha256']
        launcher=previous.read('test-kv-delta-http.bat').decode().replace('\r\n','\n');assert launcher.count('probe_kv_delta_http.py')==1
        launcher=launcher.replace('probe_kv_delta_http.py','probe_kv_delta_qpc_http.py').encode()
        assert launcher==z.read(methods+'test_kv_delta_qpc_http.bat')==z.read('test-kv-delta-qpc-http.bat') and sha(launcher)==prep['launcher_sha256']
        baseline=previous.read(methods+'verify_kv_delta_http.py').decode().replace('\r\n','\n')
        current=Path(__file__).read_text(encoding='utf8')
        for block in prep['verifier_retained_blocks']:
            a=baseline.index(block['begin']);b=baseline.index(block['end'],a);part=baseline[a:b]
            assert sha(part.encode())==block['sha256'] and current.count(part)==1
        for n,h in prep['files'].items():
            p=methods+n
            if p in previous.namelist():assert sha(previous.read(p))==prep['modified_helper_sources'].get(n,{}).get('before_sha256',h)
            else:assert n in prep['newly_recorded_helper_sources']
    ledger=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').decode().splitlines()];assert ledger[-1]['event']=='finished' and ledger[-1]['exit_code']==workflow['exit_code']
    if workflow['exit_code']!=0:
        result={'status':'compact_kv_delta_qpc_http_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),'performance_admitted':False,'scope':'Raw terminal failed pilot/source lineage preserved; no quality/timing admission.'}
        (out/(pilot+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result));raise SystemExit(0)
    assert workflow['status']=='compact_http_finished_needs_independent_replay'
    quality = doc(native+'quality.json')
    assert quality['status'] == 'passed' and len(quality['cases']) == 20 and len(quality['stream_cases']) == 2
    for case in quality['cases']:
        assert case['exact']
        expected = doc('results/native-mtp3-http/'+case['oracle']+'-response.json')['choices'][0]
        actual = doc(native+case['key']+'-response.json')['choices'][0]
        for field in ('prompt_token_ids', 'response_token_ids'): assert actual[field] == expected[field]
    regression_raw=z.read('results/batch-http-serialization-regression.json')
    regression=json.loads(regression_raw)
    assert sha(regression_raw)==quality['regression_fixture_sha256']
    records={(v['repeat'],v['row']):v for v in quality['regression_cases']}
    assert set(records)=={(0,0),(0,1),(1,0),(1,1)}
    with zipfile.ZipFile(out/'batch-http-pilot-attempt1-evidence.zip') as failed:
        assert regression['source_archive_sha256']==sha((out/'batch-http-pilot-attempt1-evidence.zip').read_bytes())
        old_tokens={}
        for line in failed.read('results/batch-http-pilot-attempt1/native/worker/worker-stdout.jsonl').decode().splitlines():
            try:event=json.loads(json.loads(line)['line'])
            except ValueError:continue
            if event.get('event')=='token':old_tokens.setdefault(event['id'],[]).append(event['token_id'])
        for row,case in enumerate(regression['cases']):
            path=case['source_request_path']
            assert json.loads(failed.read(path))==case['body']
            old_events=[json.loads(line)['data'] for line in failed.read(path.removesuffix('-request.json')+'-events.jsonl').decode().splitlines()]
            old_ids={json.loads(v)['id'] for v in old_events if v!='[DONE]' and 'id' in json.loads(v)}
            assert len(old_ids)==1 and old_tokens[old_ids.pop()]==case['old_token_prefix']
            actuals=[]
            for repeat in (0,1):
                stem=native+'serialization-regression-'+str(repeat)+'-'+str(row)
                body=doc(stem+'-request.json');expected_body=dict(case['body']);expected_body['stream']=False
                assert body==expected_body
                actual=doc(stem+'-response.json')['choices'][0]
                assert actual['response_token_ids'][:len(case['old_token_prefix'])]==case['old_token_prefix']
                assert len(actual['response_token_ids'])==128 and len(actual['prompt_token_ids'])==104
                assert records[(repeat,row)]['old_prefix_exact'] and records[(repeat,row)]['full_ids_repeat_exact']==bool(repeat)
                actuals.append(actual)
            for field in ('prompt_token_ids','response_token_ids'):assert actuals[0][field]==actuals[1][field]
    repair=doc('results/unicode-transport-repair2-verified.json')
    assert repair['status']=='canonical_unicode_transport_repair_independently_verified'
    assert repair['all_repaired_bytes_match_cpython']
    assert sha(z.read('Garnet/test2026/cuda/qwen38_5090/native_worker_io.py'))==repair['repair_source_sha256']
    workers = {}
    tokens = {}
    ready = []
    for line in z.read(native+'worker/worker-stdout.jsonl').decode().splitlines():
        try: event = json.loads(json.loads(line)['line'])
        except ValueError: continue
        kind = event.get('event')
        if kind == 'ready': ready.append(event)
        elif kind == 'token':
            key = event['id']; assert key not in workers
            tokens.setdefault(key, []).append(event['token_id'])
            assert event['sampled_tokens'] == len(tokens[key])
        elif kind == 'completed':
            key = event['id']; assert key not in workers and event['generated_ids'] == tokens[key]
            workers[key] = event
        elif kind == 'error': raise AssertionError(event)
    assert len(ready) == 1 and ready[0]['batch'] == 4 and ready[0]['library_sha256'] == evidence['library_sha256']
    dispatch = [json.loads(line) for line in z.read(native+'worker/batch-dispatch.jsonl').decode().splitlines()]
    dispatched = []
    for group in dispatch:
        assert group['physical_capacity'] == 4 and group['real_rows'] == len(group['request_ids']) in (1,2,3,4)
        dispatched.extend(group['request_ids'])
    assert len(dispatched) == len(set(dispatched)) and set(dispatched) == set(workers)
    assert any(group['real_rows'] == 4 for group in dispatch)
    group_for = {key: group for group in dispatch for key in group['request_ids']}
    for key, work in workers.items():
        use_mtp = group_for[key]['real_rows'] == 4
        assert work['generation_mode'] == ('verified_mtp' if use_mtp else 'greedy_target')
        if use_mtp:
            counters = work['speculative_metrics']
            assert counters['target_forward_calls'] == counters['target_verification_calls'] + counters['target_prefix_recomputes']
            assert counters['target_verification_calls'] == len(counters['iterations'])
            assert 0 <= counters['accepted_draft_tokens'] <= counters['proposed_draft_tokens']
            assert counters['mtp_prefill_calls'] > 0
    def final(path):
        events = [json.loads(line)['data'] for line in z.read(path).decode().splitlines()]
        assert events[-1] == '[DONE]'
        return next(json.loads(v) for v in reversed(events) if v != '[DONE]')
    for case in quality['stream_cases']:
        name = case['oracle']; event = final(native+'stream-'+name+'-events.jsonl')
        expected = doc('results/native-mtp3-http/'+name+'-response.json')['choices'][0]
        for field in ('prompt_token_ids','response_token_ids'): assert event[field] == expected[field]
    manifests = [name for name in z.namelist() if name.startswith(native+'benchmark/') and name.endswith('-manifest.json')]
    assert len(manifests) == 1
    manifest = doc(manifests[0]); prefix = manifests[0].rsplit('/',1)[0]+'/'
    run = verify_run(manifest, lambda name:z.read(prefix+name))
    assert len(run['cohorts']) == 8
    samples = [json.loads(line) for line in z.read(prefix+manifest['run_id']+'-vram.jsonl').decode().splitlines()]
    measured = 0
    dispatch_by_cohort = {}
    sampling_by_cohort = {}
    dispatch_index = {request_id:index for index,group in enumerate(dispatch) for request_id in group["request_ids"]}
    for cohort in manifest['cohorts']:
        assert cohort['concurrency'] in (1,2,4,8) and len(cohort['request_keys']) == 8
        phase = cohort['case']+'-c'+str(cohort['concurrency'])
        rows = [s for s in samples if s['phase'] == phase][:cohort['samples']]
        assert len(rows) == cohort['samples'] and max(s['used_bytes'] for s in rows) == cohort['sampled_peak_adapter_used_bytes']
        times = [float(sample['monotonic']) for sample in rows]
        assert all(right >= left for left, right in zip(times, times[1:]))
        sampling_by_cohort[phase] = {'samples': len(times),
            'maximum_interval_seconds': max((right-left for left,right in zip(times,times[1:])), default=0),
            'scope': 'Observed sampler interval; not a measurement of paging, eviction or GPU stall duration.'}
        cohort_native_ids=[]
        for key in cohort['request_keys']:
            event = final(prefix+key+'-events.jsonl'); work = workers[event['id']]
            cohort_native_ids.append(event['id'])
            assert event['response_token_ids'] == work['generated_ids'] and len(work['generated_ids']) == 128
            assert event['prompt_token_ids'] == work['prompt_ids']
            assert event['study']['generation_mode'] == work['generation_mode']
            body = doc(prefix+key+'-request.json')
            if cohort['case'] == 'image':
                assert sha(base64.b64decode(body['messages'][0]['content'][0]['image_url']['url'].split(',',1)[1])) == work['image_sha256']
            measured += 1
        indices={dispatch_index[key] for key in cohort_native_ids}
        groups=[dispatch[index] for index in indices]
        assert {key for group in groups for key in group['request_ids']}==set(cohort_native_ids)
        dispatch_by_cohort[phase]={'physical_capacity':4,'groups':len(groups),
            'single_real_row_groups':sum(g['real_rows']==1 for g in groups),
            'two_real_row_groups':sum(g['real_rows']==2 for g in groups),
            'three_real_row_groups':sum(g['real_rows']==3 for g in groups),
            'four_real_row_groups':sum(g['real_rows']==4 for g in groups),
            'mtp_requests':sum(workers[k]['generation_mode']=='verified_mtp' for k in cohort_native_ids),
            'greedy_requests':sum(workers[k]['generation_mode']=='greedy_target' for k in cohort_native_ids)}
    assert measured == 64
    assert workflow['grouping_clock']=='perf_counter'
    for group in dispatch:
        assert group['grouping_clock']=='perf_counter'
        start=group['grouping_started_perf_counter'];deadline=group['grouping_deadline_perf_counter'];sent=group['dispatch_perf_counter']
        assert abs(deadline-start-.010)<1e-8 and sent>=start
        if group['real_rows']<4:assert sent>=deadline
        assert set(group['request_created_perf_counter'])==set(group['queue_put_started_perf_counter'])==set(group['request_ids'])
        for key in group['request_ids']:assert group['request_created_perf_counter'][key]<=group['queue_put_started_perf_counter'][key]<=sent
    metadata=doc(native+'metadata.json');r=ready[0]
    assert metadata['grouping_clock']=='perf_counter'
    assert r['kv_delta_state'] and r['shared_output_storage'] and metadata['kv_delta_state'] and metadata['shared_output_storage']
    assert metadata['joined_mlp_projections'] and r['joined_mlp_projections'] and r['deferred_state_commit'] and metadata['deferred_state_commit']
    assert metadata['ready']==r and metadata['physical_batch']==4 and metadata['grouping_delay_ms']==10
    assert metadata['host_embedding_lookup'] and r['host_embedding_lookup'] and metadata['embedding_mode']==r['embedding_mode']=='host_mapped_bf16' and r['embedding_table_bytes']==0
    assert metadata['mtp_depth']==r['mtp_depth']==3 and metadata['mtp_policy']==r['mtp_policy']=='four_real_rows_equal_limits_else_greedy'
    for path,h in metadata['file_sha256'].items():
        n=path.replace('\\','/').split('5090-study-20261023/',1)[1]
        if n=='build/bin/garnet.dll':assert h==evidence['library_sha256']
        else:assert sha(z.read(n))==h
    references = []
    for name in ('vllm-graph-linkfix-evidence.zip','sglang-corrected-ipv4-evidence.zip'):
        with zipfile.ZipFile(out/name) as ref:
            names = [n for n in ref.namelist() if n.endswith('-manifest.json') and n.rsplit('/',1)[-1] not in ('evidence-manifest.json','preparation-manifest.json','source-manifest.json')]
            assert len(names) == 1
            m = json.loads(ref.read(names[0])); stem = names[0].rsplit('/',1)[0]+'/'
            references.append(verify_run(m, lambda n:ref.read(stem+n)))
    assert manifest['context_capacity']==4096 and manifest['image_format']=='jpeg' and manifest['image_order']=='image-first' and manifest['image_nonce_pattern']=='binary-grid'
    for reference_name in ('vllm-graph-linkfix-evidence.zip','sglang-corrected-ipv4-evidence.zip'):
        with zipfile.ZipFile(out/reference_name) as reference_archive:
            reference_paths=[n for n in reference_archive.namelist() if n.endswith('-manifest.json') and n.rsplit('/',1)[-1] not in ('evidence-manifest.json','preparation-manifest.json','source-manifest.json')]
            assert len(reference_paths)==1
            reference_manifest=json.loads(reference_archive.read(reference_paths[0]))
            assert reference_manifest['gpu']==manifest['gpu'] and reference_manifest['context_capacity']==4096
    assert [r['run_id'] for r in references]==[r['run_id'] for r in parent['references']]
    comparison=compare(run,references)
    assert comparison['has_vllm_and_sglang'] and all(c['identical_request_bytes'] and c['identical_transport'] and c['identical_prompt_output_counts'] for c in comparison['cohorts'])
    parent_ratios={}
    for key,cohort in run['cohorts'].items():
        assert cohort['requests']==parent['run']['cohorts'][key]['requests']
        parent_ratios[key]={metric:cohort['metrics'][metric]/parent['run']['cohorts'][key]['metrics'][metric] for metric in ('ttft_median_ms','decode_median_tokens_per_second','output_tokens_per_second')}
    resource=doc(base+'resources/resource-monitor.json');lifetime=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert resource['gpu']==manifest['gpu']
    assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(lifetime) and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in lifetime)
    assert workflow['auxiliary_observer_during_timing'] is False
    result={'status':'compact_kv_delta_qpc_http_pilot_replayed','archive_sha256':sha(archive.read_bytes()),'library_sha256':evidence['library_sha256'],'files_verified':len(evidence['files']),
        'grouping_delay_ms':10,'grouping_clock':'perf_counter','physical_batch':4,'kv_delta_state':True,'shared_output_storage':True,'deferred_state_commit':True,'joined_mlp_projections':True,'host_embedding_lookup':True,
        'ratios_to_parent_pilot':parent_ratios,'serialization_regression_captures':4,'measured_sse_worker_pairs':measured,'actual_dispatch_by_cohort':dispatch_by_cohort,
        'dispatch_group_sizes':{str(n):sum(g['real_rows']==n for g in dispatch) for n in (1,2,3,4)},'run':run,'references':references,'comparison':comparison,'raw_cohorts':manifest['cohorts'],'sampling_by_cohort':sampling_by_cohort,
        'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':sha(z.read(methods+'verify_kv_delta_qpc_http.py')),'auxiliary_observer_during_timing':False,'clean_repeat_required':True,'two_x_established':False,
        'scope':'One compact B4/T4/MTP3 short41text/104image prompt,128output seed3 pilot with10ms grouping.64raw SSE/native pairs and quality/serialization/source/dispatch/accounting verified; matched ordinary untuned vLLM/SGLang references. Parent/native ratios descriptive, not causal stable gains. Actual4K, tuned references, broad quality and clean repeats remain required; no2x admission.'}
(out/(pilot+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ('run','references','raw_cohorts')},indent=2))
