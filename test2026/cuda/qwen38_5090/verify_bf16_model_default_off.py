"""Replay actual-model default-off IDs, state hashes and raw hidden arrays."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-model-default-off-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
checks=[];hidden=[]
with zipfile.ZipFile(archive) as z:
    doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
    for n,h in m['files'].items():assert sha(z.read(n))==h
    for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
    w=doc(base+'workflow.json');passed=w['exit_code']==0
    assert w['cleanup_finished'] and w['library_unchanged'] and not w['bf16_kv_enabled'] and w['source_sha256']==w['source_after_sha256']
    for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    for n,h in m['observed_unchanged_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
    if passed:assert m['observed_unchanged_source_sha256']==m['expected_unchanged_source_sha256']
    b=doc('results/bf16-model-integration-build-attempt1-verified.json')
    assert b['status']=='bf16_model_native_build_replayed' and b['library_sha256']==w['library_sha256']
    assert sha(z.read('results/bf16-model-integration-build-attempt1-verified.json'))==w['build_proof_sha256']
    assert b['archive_sha256']==w['build_archive_sha256']==m['linked_archives']['bf16-model-integration-build-attempt1-evidence.zip']
    events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').splitlines()]
    assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
    parent='native-kv-delta-default-off-attempt1';p=doc('results/'+parent+'-verified.json')
    assert p['status']=='native_kv_delta_default_off_replayed' and p['raw_last_hidden_all_exact']
    assert p['archive_sha256']==w['parent_archive_sha256']==m['linked_archives'][parent+'-evidence.zip']
    with zipfile.ZipFile(out/(parent+'-evidence.zip')) as previous:
        oldbase='results/'+parent+'/';old=json.loads(previous.read(oldbase+'native/report.json'))
        prior=json.loads(previous.read(oldbase+'workflow.json'))
        assert m['expected_unchanged_source_sha256']=={n:h for n,h in prior['source_sha256'].items() if n not in w['source_sha256']}
        probe=methods+'probe_b4_host_joined_mtp.py'
        assert z.read(probe)==previous.read(probe) and sha(z.read(probe))==w['probe_sha256']
        assert z.read('test-b4-host-joined-mtp.bat')==previous.read('test-b4-host-joined-mtp.bat')
        if base+'native/report.json' in m['files']:
            r=doc(base+'native/report.json');assert r['garnet_dll_sha256']==w['library_sha256']
            refs={(c['group'],c['repeat']):c for c in old['cases']}
            for c in r['cases']:
                key=(c['group'],c['repeat']);ref=refs[key]
                exact=all(c[k]==ref[k] for k in ('generated_ids','ordinary_ids','expected_ids','ordinary_state','mtp_state','limits','prompt_lengths'))
                checks.append({'group':key[0],'repeat':key[1],'retained_full_model_exact':exact})
                for mode in ('ordinary','mtp'):
                    file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32'
                    raw=z.read(base+'native/'+file);assert sha(raw)==c[mode+'_state']['last_hidden_sha256']
                    hidden.append(raw==previous.read(oldbase+'native/'+file))
                if passed:assert exact and c['unjoined_b4_ids_and_state_exact'] and c['duplicate_pair_isolation_exact'] and c['ordinary_state']==c['mtp_state'] and c['generated_ids']==c['expected_ids']==c['ordinary_ids']
            if passed:
                assert r['status']=='backend_smoke_completed' and [(c['group'],c['repeat']) for c in r['cases']]==[(0,0),(0,1),(1,0),(1,1)]
                assert len(hidden)==8 and all(hidden) and w['status']=='bf16_default_off_finished_needs_replay'
        else:assert not passed
    monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()]
    peak=max(s['used_bytes'] for s in samples)
    assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
    assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
    archived=sha(z.read(methods+'verify_bf16_model_default_off.py'))
r={'status':'bf16_model_default_off_replayed' if passed else 'bf16_model_default_off_failure_preserved','archive_sha256':sha(archive.read_bytes()),
   'library_sha256':w['library_sha256'],'files_verified':len(m['files']),'cases':checks,'raw_last_hidden_arrays':len(hidden),'raw_last_hidden_all_exact':bool(hidden) and all(hidden),
   'sampled_peak_adapter_used_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
   'bf16_model_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
