"""Replay new-DLL B2 retained IDs and complete target-state hash witnesses."""
from pathlib import Path,PureWindowsPath
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);mode=sys.argv[2];assert mode in ('device','host')
name='shared-mtp3-full-b2-host-embedding-'+mode+'-attempt1';archive=out/(name+'-evidence.zip')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;evidence=json.loads(z.read('evidence-manifest.json'))
    assert len(z.namelist())==len(set(z.namelist()))==len(evidence['files'])+1
    for path,digest in evidence['files'].items():assert sha(z.read(path))==digest,path
    for path,digest in evidence['linked_archives'].items():assert sha((out/path).read_bytes())==digest,path
    doc=lambda path:json.loads(z.read(path));base='results/'+name+'/'
    workflow=doc(base+'workflow.json');passed=workflow['exit_code']==0
    report=doc(base+'native/report.json') if base+'native/report.json' in evidence['files'] else None
    build=doc('results/native-host-embedding-build-attempt1/manifest.json')
    assert build['status']=='native_host_embedding_build_finished_unqualified' and build['exit_code']==0
    assert build['library_sha256']==evidence['library_sha256']==workflow['library_sha256']
    assert workflow['mode']==mode and workflow['host_embedding_lookup']==(mode=='host')
    for key in ('candidate_source_sha256','unchanged_policy_source_sha256'):
        for path,digest in build[key].items():assert sha(z.read('Garnet/'+path))==digest
    rows=doc('results/host-embedding-rows-attempt1-verified.json')
    assert rows['status']=='host_embedding_selected_rows_and_build_evidence_verified'
    assert rows['library_sha256']==evidence['library_sha256'] and rows['raw_arrays_verified']==226 and rows['rejections_verified']==22
    assert rows['archive_sha256']==evidence['linked_archives']['host-embedding-rows-attempt1-evidence.zip']
    prior=doc('results/shared-mtp3-full-b2-deferred-attempt1-verified.json')
    assert prior['model_gate_passed'] and prior['strict_target_state_hash_gate_passed']
    assert prior['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b2-deferred-attempt1-evidence.zip']
    assert prior['library_sha256']==build['old_library_sha256']!=evidence['library_sha256']
    old_state_path='results/shared-mtp3-full-b2-deferred-attempt1/native/report.json'
    old_raw=z.read(old_state_path);old=doc(old_state_path);old_cases={(c['group'],c['repeat']):c for c in old['cases']}
    assert len(old_cases)==4 and old['status']=='backend_smoke_completed'
    assert workflow['retained_state_report_sha256']==sha(old_raw)
    http_raw=z.read('results/batch-mtp-deferred-http-pilot-attempt1-verified.json');http=json.loads(http_raw)
    assert sha(http_raw)==build['prior_http_proof_sha256']
    assert http['archive_sha256']==evidence['linked_archives']['batch-mtp-deferred-http-pilot-attempt1-evidence.zip']
    with zipfile.ZipFile(out/'shared-mtp3-full-b2-deferred-attempt1-evidence.zip') as prior_zip, zipfile.ZipFile(out/'batch-mtp-deferred-http-pilot-attempt1-evidence.zip') as http_zip:
        assert prior_zip.read(old_state_path)==old_raw
        for path,digest in workflow['workspace_source_sha256'].items():
            raw=z.read('Garnet/'+path);assert sha(raw)==digest
            source=http_zip if path.endswith('/native_prompt.py') else prior_zip
            assert raw==source.read('Garnet/'+path)
        expected_source=prior_zip.read('Garnet/test2026/cuda/qwen38_5090/probe_shared_mtp3_full_b2_deferred.py').decode('utf-8-sig').replace('\r\n','\n')
    expected_source=expected_source.replace('"""Real packed B2/B4 backend qualification against saved B1 IDs; not HTTP timing."""','"""New-DLL B2 regression with host lookup off/on; exact IDs and retained state hashes."""',1)
    expected_source=expected_source.replace("path=out/'report.json'", "mode=sys.argv[4];assert mode in ('device','host');host_lookup=mode=='host'\npath=out/'report.json'",1)
    expected_source=expected_source.replace("report={'status':'starting'", "prior_path=root/'results/shared-mtp3-full-b2-deferred-attempt1/native/report.json'\nprior_raw=prior_path.read_bytes();prior=json.loads(prior_raw)\nassert prior['status']=='backend_smoke_completed'\nprior_cases={(c['group'],c['repeat']):c for c in prior['cases']}\nreport={'host_embedding_lookup':host_lookup,'retained_state_report_sha256':hashlib.sha256(prior_raw).hexdigest(),'status':'starting'",1)
    expected_source=expected_source.replace('retain_linear_inputs=True,deferred_state_commit=True)', 'retain_linear_inputs=True,deferred_state_commit=True,host_embedding_lookup=host_lookup)',1)
    expected_source=expected_source.replace(" pool=json.loads(decoder.io.memory_json()); report['external_compressed_pool']=pool", " pool=json.loads(decoder.io.memory_json()); report['external_compressed_pool']=pool\n assert pool['embedding_mode']==('host_mapped_bf16' if host_lookup else 'device_bf16')\n assert pool['embedding_table_bytes']==(0 if host_lookup else 2542796800)\n assert pool['host_embedding_mapped_tensor_bytes']==(2542796800 if host_lookup else 0)",1)
    expected_source=expected_source.replace("   report['cases'].append(result);path.write_text(json.dumps(report,indent=2))", "   retained=prior_cases[(group_index,repeat)]\n   result['retained_old_library_state_exact']=ordinary_state==retained['ordinary_state'] and mtp_state==retained['mtp_state']\n   report['cases'].append(result);path.write_text(json.dumps(report,indent=2))\n   assert result['retained_old_library_state_exact'],'New DLL state differs from retained3c3207 full-model witnesses'",1)
    expected_source=expected_source.replace("'scope':'Research B2 MTP3 packed mixed backend", "'scope':'Host lookup off/on new-DLL research B2 MTP3 packed mixed backend",1)
    probe_path='Garnet/test2026/cuda/qwen38_5090/probe_shared_mtp3_full_b2_host_embedding.py'
    assert z.read(probe_path).decode('utf-8-sig').replace('\r\n','\n')==expected_source
    assert workflow['probe_sha256']==sha(z.read(probe_path))
    if mode=='host':
        device=doc('results/shared-mtp3-full-b2-host-embedding-device-attempt1-verified.json')
        assert device['status']=='host_embedding_full_b2_device_regression_verified' and device['model_gate_passed'] and device['strict_target_state_hash_gate_passed'] and device['retained_old_library_state_exact']
        assert device['library_sha256']==evidence['library_sha256']
        assert device['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b2-host-embedding-device-attempt1-evidence.zip']
    launcher=z.read('test-shared-mtp3-full-b2-host-embedding.bat').decode('utf-8-sig')
    for flag in ('GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M32_CANONICAL'):assert 'set '+flag+'=1' in launcher
    keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green']
    teachers={key:(doc('results/native-mtp3-http/'+key+'-response.json')['choices'][0],doc('results/native-mtp3-http/'+key+'-request.json')) for key in keys}
    comparisons=[];state_checks=[]
    if report is not None:
        assert report['garnet_dll_sha256']==evidence['library_sha256']
        assert report['host_embedding_lookup']==(mode=='host') and report['retained_state_report_sha256']==sha(old_raw)
        assert report['batch']==2 and report['capacity']==4096 and report['chunk']==16
        assert report['mtp_depth']==3 and report['draft_chunk']==4 and report['draft_builder_workspace_bytes']==128<<20
        assert report['deferred_state_commit'] and workflow['deferred_state_commit'] and workflow['draft_builder_workspace_bytes']==128<<20
        for oracle in report['oracles']:
            for path,digest in oracle['oracle_sha256'].items():
                relative=PureWindowsPath(path).relative_to(PureWindowsPath('D:/CantorAI/5090-study-20261023')).as_posix()
                assert sha(z.read(relative))==digest
        if passed:
            pool=report['external_compressed_pool'];assert report['external_compressed_weights']
            assert pool['compressed_projection_pool_entries']==400 and pool['compressed_nvfp4_entries']==192 and pool['compressed_fp8_entries']==208
            assert pool['embedding_mode']==('host_mapped_bf16' if mode=='host' else 'device_bf16')
            assert pool['embedding_table_bytes']==(0 if mode=='host' else 2542796800)
            assert pool['host_embedding_mapped_tensor_bytes']==(2542796800 if mode=='host' else 0)
            assert pool['head_precision']=='modelopt_nvfp4_w4a4'
            physical=pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']
        for case in report['cases']:
            pair=keys[:2] if case['group']==0 else keys[2:];assert case['keys']==pair and case['repeat'] in (0,1)
            expected=[teachers[key][0]['response_token_ids'] for key in pair]
            assert case['expected_ids']==expected and case['prompt_lengths']==[len(teachers[key][0]['prompt_token_ids']) for key in pair]
            assert case['limits']==[teachers[key][1]['max_tokens'] for key in pair]
            retained=old_cases[(case['group'],case['repeat'])]
            ordinary=case['ordinary_state'];mtp=case['mtp_state']
            same=ordinary==mtp;legacy=ordinary==retained['ordinary_state'] and mtp==retained['mtp_state']
            assert case['full_target_state_exact']==same and case['retained_old_library_state_exact']==legacy
            for state in (ordinary,mtp):
                assert state['state_words_per_row']>0 and len(state['state_row_sha256'])==2 and len(state['slots'])==2
                assert all(len(h)==64 and set(h)<=set('0123456789abcdef') for h in state['state_row_sha256']+[state['last_hidden_sha256']])
            state_checks.append({'group':case['group'],'repeat':case['repeat'],'ordinary_mtp_exact':same,'retained_old_library_exact':legacy})
            assert case['exact_b1_ids_match']==(case['generated_ids']==expected)
            for key,actual,reference in zip(pair,case['generated_ids'],expected):
                comparisons.append({'key':key,'repeat':case['repeat'],'generated_count':len(actual),'expected_count':len(reference),'exact':actual==reference})
            if passed:
                assert case['generated_ids']==case['ordinary_ids']==expected and same and legacy
                assert case['mtp_memory']['target_prefill_chunk']==16 and case['mtp_memory']['draft_chunk']==4
                assert case['mtp_memory']['target_verifier']['deferred_state_commit'] and case['mtp_memory']['target_verifier']['canonical_state_backup_bytes']==0
                profiles=case['decoder_memory']['packed_profiles'];assert profiles['deferred_state_commit'] and profiles['host_embedding_lookup']==(mode=='host')
                counters=case['speculative_metrics'];assert counters['target_forward_calls']==counters['target_verification_calls']+counters['target_prefix_recomputes']
                assert counters['target_verification_calls']==len(counters['iterations']) and 0<=counters['accepted_draft_tokens']<=counters['proposed_draft_tokens']
                bodies=[g for g in case['decoder_memory']['graphs'] if g['native_nvfp4_projection_weights']==192 and g['native_fp8_projection_weights']==208]
                assert len(bodies)==1
                for graph in bodies:
                    assert graph['external_bf16_weight_inputs']==0 and graph['external_compressed_weight_inputs']==400 and graph['external_compressed_weight_bytes']==physical
                    assert graph['checkpoint_weight_precision']=='modelopt_mixed' and graph['graph_capture_failures']==0 and graph['graph_captures']>=1 and graph['graph_replays']>=1
        if passed:
            assert report['status']=='backend_smoke_completed' and len(report['cases'])==4
            assert {(c['group'],c['repeat']) for c in report['cases']}=={(0,0),(0,1),(1,0),(1,1)}
            assert len(comparisons)==8 and all(c['exact'] for c in comparisons)
        else:assert report['status']=='failed'
    else:assert not passed
    events=[json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
    assert events[0]['event']=='started' and events[0]['argv'][-1]==mode
    assert events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
    resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
    assert resource['child_exit_code']==workflow['exit_code'] and resource['sampled_memory']['samples']==len(samples)
    peak=max(sample['used_bytes'] for sample in samples);assert peak==resource['sampled_memory']['sampled_peak_adapter_used_bytes']
result={'status':('host_embedding_full_b2_'+mode+'_regression_verified') if passed else ('host_embedding_full_b2_'+mode+'_failed_evidence_verified'),
    'model_gate_passed':passed,'strict_target_state_hash_gate_passed':passed,'retained_old_library_state_exact':passed,
    'library_sha256':evidence['library_sha256'],'archive_sha256':sha(archive.read_bytes()),'files_verified':len(evidence['files']),
    'host_embedding_lookup':mode=='host','comparisons':comparisons,'state_witness_checks':state_checks,'sampled_peak_adapter_used_bytes':peak,
    'raw_target_state_arrays_retained':False,'loaded_module_path_observed':False,'performance_admitted':False,
    'scope':'B2 T16/draftT4 MTP3 4096 capacity: bounded retained-ID and producer-observed ordinary/MTP/previous-DLL full-state hash witnesses. Not independent raw-state arithmetic, B1 full regression, sanitizer/allocation-failure coverage, stream/HTTP/broad quality or performance admission.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
