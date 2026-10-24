"""Replay default-off new-DLL evidence against retained full-model state/IDs."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='native-kv-delta-default-off-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json')
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    base='results/'+name+'/';w=doc(base+'workflow.json');passed=w['exit_code']==0
    assert w['cleanup_finished'] and w['library_unchanged'] and not w['canonical_kv_overlay_enabled'] and not w['kv_delta_commit_enabled']
    assert w['source_sha256']==w['source_after_sha256'] and w['library_sha256']==m['library_sha256']
    for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    methods='Garnet/test2026/cuda/qwen38_5090/';plan=doc(methods+'evidence/native-kv-delta-source-handoff.json')
    buildbase='results/native-kv-delta-build-attempt1/';buildraw=z.read(buildbase+'manifest.json');b=json.loads(buildraw);receipt=doc(buildbase+'source-receipt.json')
    assert sha(buildraw)==w['build_manifest_sha256'] and b['status']=='native_kv_delta_build_finished_unqualified' and b['exit_code']==0
    assert b['library_sha256']==m['library_sha256']==sha(z.read('build/bin/garnet-native-kv-delta-attempt1.dll')) and b['library_sha256']!=b['old_library_sha256']==plan['original_library_sha256']
    assert b['candidate_source_sha256']==receipt['candidate_source_sha256']==plan['candidate_source_sha256']
    assert b['handoff_sha256']==sha(z.read(methods+'evidence/native-kv-delta-source-handoff.json'))
    for n,h in b['candidate_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in b['original_source_sha256'].items():
        raw=z.read(buildbase+'original/'+n);assert sha(raw)==h==receipt['original_source_sha256'][n]
        assert sha(raw.replace(b'\r\n',b'\n'))==plan['base_normalized_sha256'][n]
    assert sha(z.read('results/native-kv-delta-source.zip'))==plan['source_zip_sha256']
    parent='shared-mtp3-full-b4-t4-host-joined-attempt1';proof=doc('results/'+parent+'-verified.json')
    assert proof['status']=='b4_host_joined_mtp_correctness_replayed' and proof['model_gate_passed'] and proof['raw_last_hidden_all_exact']
    assert proof['library_sha256']==b['old_library_sha256'] and proof['archive_sha256']==w['parent_archive_sha256']==m['linked_archives'][parent+'-evidence.zip']
    operator='canonical-kv-overlay-plugin-attempt1';op=doc('results/'+operator+'-verified.json')
    assert op['status']=='canonical_kv_overlay_operator_replayed' and op['raw_context_arrays']==96 and len(op['cases'])==24
    assert op['archive_sha256']==plan['operator_archive_sha256']==b['operator_archive_sha256']==m['linked_archives'][operator+'-evidence.zip']
    with zipfile.ZipFile(out/(operator+'-evidence.zip')) as qualified:
        for n,recipe in plan['qualified_source_derivation'].items():
            text=qualified.read(methods+recipe['source']).decode()
            for before,after in recipe['changes']:assert text.count(before)==1;text=text.replace(before,after)
            assert z.read('Garnet/'+n)==text.encode()
    checks=[];hidden=[];peak=None
    with zipfile.ZipFile(out/(parent+'-evidence.zip')) as previous:
        oldbase='results/'+parent+'/';oldreport=json.loads(previous.read(oldbase+'native/report.json'))
        probe=methods+'probe_b4_host_joined_mtp.py'
        assert z.read(probe)==previous.read(probe) and sha(z.read(probe))==w['probe_sha256']
        assert z.read('test-b4-host-joined-mtp.bat')==previous.read('test-b4-host-joined-mtp.bat')
        report_path=base+'native/report.json'
        if report_path in m['files']:
            report=doc(report_path);assert report['garnet_dll_sha256']==m['library_sha256']
            assert report['batch']==4 and report['capacity']==4096 and report['chunk']==report['draft_chunk']==4
            assert report['joined_mlp_projections'] and report['host_embedding_lookup'] and report['deferred_state_commit'] and report['draft_builder_workspace_bytes']==256<<20
            oldcases={(c['group'],c['repeat']):c for c in oldreport['cases']}
            for c in report['cases']:
                key=(c['group'],c['repeat']);ref=oldcases[key]
                exact=all(c[k]==ref[k] for k in ('generated_ids','ordinary_ids','expected_ids','ordinary_state','mtp_state','limits','prompt_lengths'))
                same=c['ordinary_state']==c['mtp_state'];ids=c['generated_ids']==c['ordinary_ids']==c['expected_ids']
                checks.append({'group':key[0],'repeat':key[1],'retained_full_model_exact':exact,'same_profile_state_exact':same,'ids_exact':ids})
                for mode in ('ordinary','mtp'):
                    file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32'
                    raw=z.read(base+'native/'+file);hidden.append(raw==previous.read(oldbase+'native/'+file));assert sha(raw)==c[mode+'_state']['last_hidden_sha256']
                if passed:
                    assert exact and same and ids and c['unjoined_b4_ids_and_state_exact'] and c['duplicate_pair_isolation_exact']
                    memory=c['decoder_memory'];p=memory['packed_profiles']
                    assert p['joined_mlp_projections'] and p['host_embedding_lookup'] and p['deferred_state_commit'] and not p.get('kv_delta_state',False)
                    assert memory['compressed_projection_pool_entries']==336
                    bodies=[g for g in memory['graphs'] if g['native_nvfp4_projection_weights']==128 and g['native_fp8_projection_weights']==208]
                    assert len(bodies)==1 and bodies[0]['external_compressed_weight_inputs']==336 and bodies[0]['external_bf16_weight_inputs']==0 and bodies[0]['graph_capture_failures']==0
            if passed:
                assert [(c['group'],c['repeat']) for c in report['cases']]==[(0,0),(0,1),(1,0),(1,1)]
                assert w['status']=='default_off_finished_needs_independent_replay' and report['status']=='backend_smoke_completed' and len(checks)==4 and len(hidden)==8 and all(hidden)
        else:assert not passed
        if base+'resources/resource-monitor.json' in m['files']:
            monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
            assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
            peak=max(s['used_bytes'] for s in samples);assert monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
            assert monitor['gpu']==json.loads(previous.read(oldbase+'resources/resource-monitor.json'))['gpu']
        else:assert not passed
    for prefix in (buildbase+'build-ledger/',base+'ledger/'):
        events=[json.loads(l) for l in z.read(prefix+'commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished'
        if passed:assert events[-1]['exit_code']==0
    verifier=sha(z.read(methods+'verify_native_kv_delta_default_off.py'))
result={'status':'native_kv_delta_default_off_replayed' if passed else 'native_kv_delta_default_off_failure_preserved','archive_sha256':sha(archive.read_bytes()),
        'library_sha256':m['library_sha256'],'files_verified':len(m['files']),'cases':checks,'raw_last_hidden_arrays_compared':len(hidden),'raw_last_hidden_all_exact':bool(hidden) and all(hidden),
        'sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':verifier,
        'compact_model_admitted':False,'final_write_helper_admitted':False,'performance_admitted':False,
        'scope':'Fresh new native DLL with optimization disabled. Same unchanged joined-host B4/T4 full-model raw IDs/last-hidden and state hash witnesses compared against retained qualified model. Full raw state arrays absent. Compact path/final-write helper, stream, HTTP, actual4K, broad quality, speed and2x remain unqualified.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
