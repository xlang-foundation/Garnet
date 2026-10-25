"""Replay BF16 model/helper fixed-corpus qualification; never a speed test."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-model-b4-t32-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
checks=[];hidden=[]
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    w=doc(base+'workflow.json');passed=w['exit_code']==0
    assert w['cleanup_finished'] and w['library_unchanged'] and w['bf16_kv_enabled'] and w['source_sha256']==w['source_after_sha256']
    for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    prep_raw=z.read(methods+'evidence/bf16-model-b4-t32-preparation.json');p=json.loads(prep_raw);assert sha(prep_raw)==w['preparation_sha256']
    for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
    for n,h in p['pinned_inputs'].items():assert sha(z.read(n))==h
    assert sha(z.read('results/bf16-kv-metadata-fixture.json'))==p['fixture_sha256']==w['fixture_sha256']
    assert p['model_config_sha256']==w['model_config_sha256']==sha(z.read('models/Qwen3.8-27B-NVFP4/config.json'))=='71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8'
    recipe=doc(methods+'evidence/bf16-model-b4-t32-probe-derivation.json');raw=z.read(methods+recipe['source']);assert sha(raw)==recipe['source_sha256']
    text=raw.decode().replace('\r\n','\n')
    for before,after in recipe['changes']:assert text.count(before)==1;text=text.replace(before,after)
    assert text.encode()==z.read(methods+'probe_bf16_model_b4_t32.py') and sha(text.encode())==recipe['result_sha256']==w['probe_sha256']
    build=doc('results/bf16-model-integration-build-attempt1-verified.json');parent=doc('results/bf16-model-default-off-attempt1-verified.json')
    assert build['status']=='bf16_model_native_build_replayed' and parent['status']=='bf16_model_default_off_replayed' and parent['raw_last_hidden_all_exact']
    assert w['library_sha256']==build['library_sha256']==parent['library_sha256']
    assert w['build_archive_sha256']==build['archive_sha256']==m['linked_archives']['bf16-model-integration-build-attempt1-evidence.zip']
    assert w['parent_archive_sha256']==parent['archive_sha256']==m['linked_archives']['bf16-model-default-off-attempt1-evidence.zip']
    assert sha(z.read('results/bf16-model-integration-build-attempt1-verified.json'))==w['build_proof_sha256']
    events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').splitlines()]
    assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
    if base+'native/report.json' in m['files']:
        r=doc(base+'native/report.json');assert r['garnet_dll_sha256']==w['library_sha256']
        assert r['bf16_kv_state'] and not r['kv_delta_state'] and r['batch']==4 and r['capacity']==4096 and r['prefill_chunk']==32 and r['chunk']==4 and r['register_delta_128']
        with zipfile.ZipFile(out/'native-kv-delta-default-off-attempt1-evidence.zip') as previous:
            oldbase='results/native-kv-delta-default-off-attempt1/';oldraw=previous.read(oldbase+'native/report.json');old=json.loads(oldraw)
            assert sha(oldraw)==r['retained_unjoined_b4_report_sha256']==p['pinned_inputs'][oldbase+'native/report.json']
            refs={(c['group'],c['repeat']):c for c in old['cases']}
            for c in r['cases']:
                key=(c['group'],c['repeat']);ref=refs[key]
                ids=c['generated_ids']==c['ordinary_ids']==c['expected_ids']==ref['generated_ids']
                state=c['ordinary_state']==c['mtp_state'];checks.append({'group':key[0],'repeat':key[1],'ids_exact':ids,'ordinary_mtp_state_exact':state})
                for mode in ('ordinary','mtp'):
                    file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32';raw=z.read(base+'native/'+file)
                    assert len(raw)==4*5120*4 and sha(raw)==c[mode+'_state']['last_hidden_sha256']
                    hidden.append(raw)
                if passed:
                    assert ids and state and c['retained_fp32_reference_ids_exact'] and c['duplicate_pair_isolation_exact'] and hidden[-1]==hidden[-2]
                    assert c['ordinary_state']['state_words_per_row']==106823680 and len(c['ordinary_state']['state_row_sha256'])==4
                    mem=c['decoder_memory'];profile=mem['packed_profiles']
                    assert profile['bf16_kv_state'] and profile['kv_cache_dtype']=='bf16' and profile['prefill']==32 and profile['decode']==4 and not profile['kv_delta_state']
                    bodies=[g for g in mem['graphs'] if g['native_nvfp4_projection_weights']==128 and g['native_fp8_projection_weights']==208]
                    assert len(bodies)==2 and all(g['external_compressed_weight_inputs']==336 and g['external_bf16_weight_inputs']==0 and g['graph_capture_failures']==0 for g in bodies)
        if passed:
            assert r['status']=='backend_smoke_completed' and [(c['group'],c['repeat']) for c in r['cases']]==[(0,0),(0,1),(1,0),(1,1)]
            assert w['status']=='bf16_b4_t32_finished_needs_replay' and len(hidden)==8
            gate=r['runtime_metadata_gate'];assert gate=={'valid_profiles':4,'rebuild_rejections':4,'invalid_layouts':8856,'fixture_sha256':p['fixture_sha256']}
            negative=r['negative_commit_gate'];assert negative['status']=='passed' and negative['before']==negative['after'] and negative['valid_zero_prefix_modes']==3 and len(negative['rejected_calls'])==16
            contract=r['kv_delta_contract'];assert contract['canonical_state_width']==106823680 and contract['target_output_width']==43030528 and contract['linear_replay_width']==39714816 and contract['strategies']==[1,1]
            assert contract['commit']['ready'] and not contract['commit']['poisoned'] and contract['commit']['commits']==2
    else:assert not passed
    monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples)
    assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
    assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
    archived=sha(z.read(methods+'verify_bf16_model_b4_t32.py'))
r={'status':'bf16_model_b4_t32_fixed_corpus_replayed' if passed else 'bf16_model_b4_t32_failure_preserved','archive_sha256':sha(archive.read_bytes()),
   'library_sha256':w['library_sha256'],'files_verified':len(m['files']),'cases':checks,'raw_last_hidden_arrays':len(hidden),'ordinary_mtp_raw_hidden_exact':bool(hidden) and all(hidden[i]==hidden[i+1] for i in range(0,len(hidden),2)),
   'sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
   'broad_quality_qualified':False,'b8_qualified':False,'actual_4k_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
