"""Replay compact model IDs/raw hidden and state witnesses against default-off."""
from pathlib import Path
import ast,hashlib,io,json,sys,zipfile
out=Path(sys.argv[1]);name='kv-delta-full-b4-t4-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';w=doc(base+'workflow.json');passed=w['exit_code']==0
    assert w['cleanup_finished'] and w['library_unchanged'] and w['library_sha256']==m['library_sha256'] and w['kv_delta_state'] and w['shared_output_storage']
    assert w['candidate_source_sha256']==w['source_after_sha256'] and w['unchanged_native_source_sha256']==w['native_after_sha256'] and w['unchanged_parent_source_sha256']==w['parent_source_after_sha256']
    for group in ('candidate_source_sha256','unchanged_native_source_sha256','unchanged_parent_source_sha256'):
        for n,h in w[group].items():assert sha(z.read('Garnet/'+n))==h
    methods='Garnet/test2026/cuda/qwen38_5090/';planraw=z.read(methods+'evidence/kv-delta-model-source-handoff.json');plan=json.loads(planraw)
    assert sha(planraw)==w['source_plan_sha256'] and plan['native_library_sha256']==m['library_sha256']
    assert sha(z.read(methods+'run_kv_delta_full_b4_t4.py'))==plan['runner_sha256']
    assert plan['candidate_source_sha256']==w['candidate_source_sha256'] and not plan['timing_enabled'] and not plan['gpu_run']
    with zipfile.ZipFile(io.BytesIO(z.read('results/kv-delta-model-source.zip'))) as staged:
        assert sha(z.read('results/kv-delta-model-source.zip'))==plan['source_zip_sha256']
        assert set(staged.namelist())==set(plan['candidate_source_sha256'])
        for n,h in plan['candidate_source_sha256'].items():assert sha(staged.read(n))==h and staged.read(n)==z.read('Garnet/'+n)
    for n,h in w['original_source_sha256'].items():
        raw=z.read(base+'original/'+n);assert sha(raw)==h
        assert sha(raw.replace(b'\r\n',b'\n')) in (plan['base_normalized_sha256'].get(n),sha(z.read('Garnet/'+n).replace(b'\r\n',b'\n')))
    cpu=doc(methods+'evidence/kv-delta-model-cpu-tests.json');assert cpu['status']=='compact_model_cpu_contracts_passed' and cpu['exit_code']==0 and cpu['test_count']==27 and 'Ran 27 tests' in cpu['raw_output_utf8'] and not cpu['gpu_run']
    for n,h in cpu['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    parent='native-kv-delta-default-off-attempt1';proof=doc('results/'+parent+'-verified.json')
    assert proof['status']=='native_kv_delta_default_off_replayed' and proof['raw_last_hidden_all_exact'] and proof['raw_last_hidden_arrays_compared']==8
    assert proof['library_sha256']==m['library_sha256'] and proof['archive_sha256']==w['parent_archive_sha256']==plan['default_off_archive_sha256']==m['linked_archives'][parent+'-evidence.zip']
    recipe=doc(methods+'evidence/kv-delta-full-probe-preparation.json');checks=[];hidden=[];peak=None;negative=None
    with zipfile.ZipFile(out/(parent+'-evidence.zip')) as previous:
        oldbase='results/'+parent+'/';oldreport=json.loads(previous.read(oldbase+'native/report.json'));oldcases={(c['group'],c['repeat']):c for c in oldreport['cases']}
        assert sha(previous.read('build/bin/garnet-native-kv-delta-attempt1.dll'))==m['library_sha256']
        probe=previous.read(methods+'probe_b4_host_joined_mtp.py').decode()
        for before,after,count in recipe['probe_derivation']:assert probe.count(before)==count;probe=probe.replace(before,after)
        assert probe.encode()==z.read(methods+'probe_kv_delta_full_b4_t4.py') and sha(probe.encode())==recipe['probe_sha256']
        launcher=previous.read('test-b4-host-joined-mtp.bat').decode();assert launcher.count('probe_b4_host_joined_mtp.py')==1
        launcher=launcher.replace('probe_b4_host_joined_mtp.py','probe_kv_delta_full_b4_t4.py').encode()
        assert launcher==z.read('test-kv-delta-full-b4-t4.bat')==z.read(methods+'test_kv_delta_full_b4_t4.bat') and sha(launcher)==w['launcher_sha256']==recipe['launcher_sha256']
        def canonical_function(raw):return ast.dump(next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='decoder_state_layout'),include_attributes=False)
        program='Garnet/xModel/qwen35/vl_0_8b/cuda_packed_program.py';assert canonical_function(z.read(program))==canonical_function(previous.read(program))
        report_path=base+'native/report.json'
        if report_path in m['files']:
            report=doc(report_path);assert report['garnet_dll_sha256']==m['library_sha256'] and report['kv_delta_state'] and report['shared_output_storage']
            assert report['batch']==4 and report['capacity']==4096 and report['chunk']==report['draft_chunk']==4 and report['draft_builder_workspace_bytes']==256<<20
            assert report['retained_unjoined_b4_report_sha256']==sha(previous.read(oldbase+'native/report.json'))
            for c in report['cases']:
                key=(c['group'],c['repeat']);ref=oldcases[key]
                exact=all(c[k]==ref[k] for k in ('generated_ids','ordinary_ids','expected_ids','ordinary_state','mtp_state','limits','prompt_lengths'))
                same=c['ordinary_state']==c['mtp_state'];ids=c['generated_ids']==c['ordinary_ids']==c['expected_ids']
                checks.append({'group':key[0],'repeat':key[1],'same_DLL_default_off_exact':exact,'same_profile_state_exact':same,'ids_exact':ids})
                for mode in ('ordinary','mtp'):
                    file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32';raw=z.read(base+'native/'+file)
                    hidden.append(raw==previous.read(oldbase+'native/'+file));assert sha(raw)==c[mode+'_state']['last_hidden_sha256']
                if passed:
                    assert exact and same and ids and c['default_off_ids_and_state_exact'] and c['duplicate_pair_isolation_exact']
                    memory=c['decoder_memory'];p=memory['packed_profiles'];assert p['kv_delta_state'] and p['deferred_state_commit'] and p['shared_output_storage'] and p['host_embedding_lookup'] and p['joined_mlp_projections']
                    assert memory['shared_output_pool_bytes']==688488448 and memory['compressed_projection_pool_entries']==336
                    assert memory['kv_delta_commit']['ready'] and memory['kv_delta_commit']['owned_layout_bytes']==1792
                    bodies=[g for g in memory['graphs'] if g['native_nvfp4_projection_weights']==128 and g['native_fp8_projection_weights']==208]
                    assert len(bodies)==2 and all(g['external_compressed_weight_inputs']==336 and g['external_bf16_weight_inputs']==0 and g['graph_capture_failures']==0 for g in bodies)
                    assert c['mtp_memory']['target_verifier']['packed_replay_graph']['device_output_bytes']==635437056
            if passed:
                assert report['status']=='backend_smoke_completed' and [(c['group'],c['repeat']) for c in report['cases']]==[(0,0),(0,1),(1,0),(1,1)] and len(hidden)==8 and all(hidden)
                contract=report['kv_delta_contract'];assert contract['canonical_state_width']==173932544 and contract['target_output_width']==43030528 and contract['linear_replay_width']==39714816 and contract['strategies']==[0,1]
                negative=report['negative_commit_gate'];assert negative['status']=='passed' and negative['metadata_rejections']==recipe['metadata_rejections_planned']==136 and negative['valid_zero_prefix_modes']==recipe['valid_zero_prefix_modes_planned']==2
                labels=['length','negative-start','fractional-start','nonfinite-count','too-many','capacity-overrun','invalid-empty-start','target-canonical-overlap','linear-canonical-overlap','target-width','closed-helper']
                assert [r['label'] for r in negative['rejected_calls']]==labels and len(labels)==recipe['negative_commits_planned']==11 and all(r['error'] for r in negative['rejected_calls'])
                assert negative['before']==negative['after'] and negative['before']['slots']==[0]*4 and negative['before']['state_words_per_row']==173932544
                zero=bytes(1<<20);digest=hashlib.sha256();remaining=173932544*4
                while remaining:
                    size=min(remaining,len(zero));digest.update(zero[:size]);remaining-=size
                assert negative['before']['state_row_sha256']==[digest.hexdigest()]*4
                for stage in ('before','after'):
                    raw=z.read(base+'native/kv-negative-'+stage+'-last-hidden.f32');assert raw==bytes(4*5120*4) and sha(raw)==negative[stage]['last_hidden_sha256']
        else:assert not passed
        if base+'resources/resource-monitor.json' in m['files']:
            monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
            peak=max(s['used_bytes'] for s in samples);assert peak==monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
            assert monitor['gpu']==json.loads(previous.read(oldbase+'resources/resource-monitor.json'))['gpu']
        else:assert not passed
    events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
    if passed:assert w['status']=='compact_model_finished_needs_independent_replay'
    verifier=sha(z.read(methods+'verify_kv_delta_full_b4_t4.py'))
result={'status':'compact_kv_delta_full_model_replayed' if passed else 'compact_kv_delta_full_model_failure_preserved','archive_sha256':sha(archive.read_bytes()),'library_sha256':m['library_sha256'],
        'files_verified':len(m['files']),'cases':checks,'raw_last_hidden_arrays_compared':len(hidden),'raw_last_hidden_all_exact':bool(hidden) and all(hidden),'negative_commit_gate':negative,
        'sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':verifier,'performance_admitted':False,
        'scope':'Compact B4/T4 same-DLL default-off ordinary/MTP raw IDs/eight hidden arrays exact and full-state hash witnesses, final-write invalid/zero-prefix checks and sources. Raw full canonical state absent; no independent CUDA arithmetic on replay, stream/HTTP, actual4K, broad-quality or2x qualification.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
