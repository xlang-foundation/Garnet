"""Replay B4 T4 MTP3 IDs, raw last-hidden bytes and same-profile state witnesses."""
from pathlib import Path,PureWindowsPath
import hashlib,json,math,struct,sys,zipfile
out=Path(sys.argv[1]);name='shared-mtp3-full-b4-t4-host-embedding-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;evidence=json.loads(z.read('evidence-manifest.json'))
    assert len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for path,digest in evidence['files'].items():assert sha(z.read(path))==digest,path
    for path,digest in evidence['linked_archives'].items():assert sha((out/path).read_bytes())==digest,path
    doc=lambda path:json.loads(z.read(path));base='results/'+name+'/'
    workflow=doc(base+'workflow.json');passed=workflow['exit_code']==0
    assert workflow['physical_batch']==4 and workflow['target_chunk']==workflow['draft_chunk']==4
    assert workflow['host_embedding_lookup'] and workflow['strict_same_profile_state_required'] and workflow['cross_profile_legacy_state_diagnostic_only']
    build=doc('results/native-host-embedding-build-attempt1/manifest.json')
    assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
    assert build['library_sha256']==workflow['library_sha256']==evidence['library_sha256']
    for group in ('candidate_source_sha256','unchanged_policy_source_sha256'):
        for path,digest in build[group].items():assert sha(z.read('Garnet/'+path))==digest
    for mode in ('device','host'):
        proof=doc('results/shared-mtp3-full-b2-host-embedding-'+mode+'-attempt1-verified.json')
        assert proof['status']=='host_embedding_full_b2_'+mode+'_regression_verified' and proof['model_gate_passed'] and proof['strict_target_state_hash_gate_passed'] and proof['retained_old_library_state_exact']
        assert proof['library_sha256']==evidence['library_sha256']
        assert proof['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b2-host-embedding-'+mode+'-attempt1-evidence.zip']
    http=doc('results/batch-mtp-host-embedding-http-pilot-attempt1-verified.json')
    assert http['status']=='batch_mtp_host_embedding_http_pilot_independently_verified' and http['library_sha256']==evidence['library_sha256'] and http['host_embedding_lookup']
    assert http['archive_sha256']==evidence['linked_archives']['batch-mtp-host-embedding-http-pilot-attempt1-evidence.zip']
    old_b4=doc('results/shared-full-b4-t4-attempt1-verified.json')
    assert old_b4['status']=='shared_full_b4_t4_id_gate_verified' and old_b4['model_gate_passed']
    assert old_b4['library_sha256']==build['old_library_sha256']
    assert old_b4['archive_sha256']==evidence['linked_archives']['shared-full-b4-t4-attempt1-evidence.zip']
    row=doc('results/host-embedding-rows-attempt1-verified.json')
    assert row['status']=='host_embedding_selected_rows_and_build_evidence_verified' and row['library_sha256']==evidence['library_sha256']
    assert row['archive_sha256']==evidence['linked_archives']['host-embedding-rows-attempt1-evidence.zip']
    old_path='results/shared-mtp3-full-b2-deferred-attempt1/native/report.json';old_raw=z.read(old_path);old=doc(old_path)
    assert workflow['retained_state_report_sha256']==sha(old_raw)
    old_cases={(c['group'],c['repeat']):c for c in old['cases']};assert len(old_cases)==4
    with zipfile.ZipFile(out/'shared-mtp3-full-b2-deferred-attempt1-evidence.zip') as prior,zipfile.ZipFile(out/'batch-mtp-deferred-http-pilot-attempt1-evidence.zip') as prior_http:
        assert prior.read(old_path)==old_raw
        for path,digest in workflow['workspace_source_sha256'].items():
            raw=z.read('Garnet/'+path);assert sha(raw)==digest
            parent=prior_http if path.endswith('/native_prompt.py') else prior
            assert raw==parent.read('Garnet/'+path)
    prep=doc('Garnet/test2026/cuda/qwen38_5090/evidence/b4-t4-host-embedding-mtp3-preparation.json')
    for path,digest in prep['source_sha256'].items():
        archived='test-shared-mtp3-full-b4-t4-host-embedding.bat' if path.endswith('.bat') else 'Garnet/test2026/cuda/qwen38_5090/'+path
        assert sha(z.read(archived))==digest
    assert prep['same_profile_ordinary_mtp_state_must_match_exactly'] and prep['cross_profile_b2_t16_legacy_hashes_diagnostic_only']
    probe='Garnet/test2026/cuda/qwen38_5090/probe_shared_mtp3_full_b4_t4_host_embedding.py'
    assert workflow['probe_sha256']==sha(z.read(probe))
    launcher=z.read('test-shared-mtp3-full-b4-t4-host-embedding.bat').decode('utf-8-sig')
    assert ' 4 "%~1" host' in launcher
    for flag in ('GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M32_CANONICAL'):assert 'set '+flag+'=1' in launcher
    report=doc(base+'native/report.json') if base+'native/report.json' in evidence['files'] else None
    comparisons=[];witnesses=[];raw_count=0
    if report is not None:
        assert report['batch']==4 and report['capacity']==4096 and report['chunk']==4
        assert report['garnet_dll_sha256']==evidence['library_sha256'] and report['host_embedding_lookup']
        assert report['retained_state_report_sha256']==sha(old_raw)
        assert report['mtp_depth']==3 and report['draft_chunk']==4 and report['draft_builder_workspace_bytes']==128<<20 and report['deferred_state_commit']
        keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green']
        teachers={key:(doc('results/native-mtp3-http/'+key+'-response.json')['choices'][0],doc('results/native-mtp3-http/'+key+'-request.json')) for key in keys}
        for oracle in report['oracles']:
            for path,digest in oracle['oracle_sha256'].items():
                relative=PureWindowsPath(path).relative_to(PureWindowsPath('D:/CantorAI/5090-study-20261023')).as_posix();assert sha(z.read(relative))==digest
        if passed:
            pool=report['external_compressed_pool'];assert report['external_compressed_weights']
            assert pool['compressed_projection_pool_entries']==400 and pool['compressed_nvfp4_entries']==192 and pool['compressed_fp8_entries']==208
            assert pool['embedding_mode']=='host_mapped_bf16' and pool['embedding_table_bytes']==0 and pool['host_embedding_mapped_tensor_bytes']==2542796800
            assert pool['hidden_size']==5120 and pool['head_precision']=='modelopt_nvfp4_w4a4'
            physical=pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']
        for case in report['cases']:
            expected_keys=(keys[:2] if case['group']==0 else keys[2:])*2
            assert case['keys']==expected_keys and case['repeat'] in (0,1)
            expected=[teachers[k][0]['response_token_ids'] for k in expected_keys]
            assert case['expected_ids']==expected and case['limits']==[teachers[k][1]['max_tokens'] for k in expected_keys]
            assert case['prompt_lengths']==[len(teachers[k][0]['prompt_token_ids']) for k in expected_keys]
            retained=old_cases[(case['group'],case['repeat'])];same=case['ordinary_state']==case['mtp_state'];legacy=[];duplicates=[];hidden={}
            for mode in ('ordinary','mtp'):
                state=case[mode+'_state'];assert state['state_words_per_row']>0 and len(state['state_row_sha256'])==len(state['slots'])==4
                assert all(len(h)==64 and set(h)<=set('0123456789abcdef') for h in state['state_row_sha256']+[state['last_hidden_sha256']])
                stem='group-'+str(case['group'])+'-repeat-'+str(case['repeat'])+'-'+mode+'-last-hidden.f32'
                raw=z.read(base+'native/'+stem);assert len(raw)==4*5120*4 and sha(raw)==state['last_hidden_sha256']
                assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',raw))
                halves=[sha(raw[:2*5120*4]),sha(raw[2*5120*4:])];assert halves==state['last_hidden_pair_sha256'];raw_count+=1;hidden[mode]=raw
                old_state=retained[mode+'_state']
                legacy.append(state['state_row_sha256']==old_state['state_row_sha256']*2 and state['slots']==old_state['slots']*2 and state['state_words_per_row']==old_state['state_words_per_row'] and halves==[old_state['last_hidden_sha256']]*2)
                duplicates.append(state['state_row_sha256'][:2]==state['state_row_sha256'][2:] and state['slots'][:2]==state['slots'][2:] and raw[:2*5120*4]==raw[2*5120*4:])
            assert case['full_target_state_exact']==same and case['cross_profile_legacy_state_exact_diagnostic']==all(legacy)
            assert case['duplicate_pair_isolation_exact']==all(duplicates)
            assert case['exact_b1_ids_match']==(case['generated_ids']==expected)
            witnesses.append({'group':case['group'],'repeat':case['repeat'],'same_profile_state_exact':same,'duplicate_pairs_exact':all(duplicates),'cross_profile_legacy_exact_diagnostic':all(legacy)})
            assert len(case['generated_ids'])==4
            comparisons.extend({'key':key,'repeat':case['repeat'],'row':row,'exact':actual==reference,'generated_count':len(actual),'expected_count':len(reference)} for row,(key,actual,reference) in enumerate(zip(expected_keys,case['generated_ids'],expected)))
            if passed:
                assert case['generated_ids']==case['ordinary_ids']==expected and same and all(duplicates) and hidden['ordinary']==hidden['mtp']
                profiles=case['decoder_memory']['packed_profiles'];assert profiles['prefill']==profiles['decode']==4 and profiles['host_embedding_lookup'] and profiles['deferred_state_commit']
                assert case['mtp_memory']['target_prefill_chunk']==case['mtp_memory']['draft_chunk']==4
                assert case['mtp_memory']['target_verifier']['deferred_state_commit'] and case['mtp_memory']['target_verifier']['canonical_state_backup_bytes']==0
                metrics=case['speculative_metrics'];assert metrics['target_forward_calls']==metrics['target_verification_calls']+metrics['target_prefix_recomputes']
                assert metrics['target_verification_calls']==len(metrics['iterations']) and 0<=metrics['accepted_draft_tokens']<=metrics['proposed_draft_tokens']
                bodies=[g for g in case['decoder_memory']['graphs'] if g['native_nvfp4_projection_weights']==192 and g['native_fp8_projection_weights']==208];assert len(bodies)==1
                for graph in bodies:
                    assert graph['external_bf16_weight_inputs']==0 and graph['external_compressed_weight_inputs']==400 and graph['external_compressed_weight_bytes']==physical
                    assert graph['checkpoint_weight_precision']=='modelopt_mixed' and graph['graph_capture_failures']==0 and graph['graph_captures']>=1 and graph['graph_replays']>=1
        if passed:
            assert report['status']=='backend_smoke_completed' and len(report['cases'])==4 and len(comparisons)==16 and raw_count==8
            assert {(c['group'],c['repeat']) for c in report['cases']}=={(0,0),(0,1),(1,0),(1,1)} and all(c['exact'] for c in comparisons)
        else:assert report['status']=='failed'
    else:assert not passed
    events=[json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
    resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert resource['child_exit_code']==workflow['exit_code'] and resource['sampled_memory']['samples']==len(samples)
    peak=max(sample['used_bytes'] for sample in samples);assert peak==resource['sampled_memory']['sampled_peak_adapter_used_bytes']
result={'status':'host_embedding_mtp3_full_b4_t4_state_gate_verified' if passed else 'host_embedding_mtp3_full_b4_t4_failed_evidence_verified',
    'model_gate_passed':passed,'strict_same_profile_target_state_hash_gate_passed':passed,'library_sha256':evidence['library_sha256'],
    'archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),'comparisons':comparisons,'state_witness_checks':witnesses,
    'raw_last_hidden_arrays_verified':raw_count,'raw_full_target_state_arrays_retained':False,'sampled_peak_adapter_used_bytes':peak,'performance_admitted':False,
    'scope':'B4 T4/draftT4 host/deferred MTP3,4096capacity: retained B1 IDs, same-profile target state hash witnesses and raw last-hidden replay with duplicate pairs. Cross-profile B2 T16 hashes diagnostic only. No independent raw-full-state arithmetic, stream/HTTP/broad quality, sanitizer or speed admission.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
