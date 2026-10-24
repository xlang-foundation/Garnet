"""Replay joined B4 full-model result without relaxing retained unjoined states."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='shared-mtp3-full-b4-t4-host-joined-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'))
 assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 for n,h in manifest['linked_archives'].items():assert sha((out/n).read_bytes())==h
 doc=lambda n:json.loads(z.read(n));base='results/'+name+'/'
 workflow=doc(base+'workflow.json');passed=workflow['exit_code']==0
 assert workflow['joined_mlp_projections'] and workflow['host_embedding_lookup'] and workflow['library_sha256']==manifest['library_sha256']
 for n,h in workflow['workspace_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 assert workflow['probe_sha256']==sha(z.read('Garnet/test2026/cuda/qwen38_5090/probe_b4_host_joined_mtp.py'))
 build=doc('results/native-host-embedding-build-attempt1/manifest.json');assert build['exit_code']==0 and build['library_sha256']==manifest['library_sha256']
 for group in ('candidate_source_sha256','unchanged_policy_source_sha256'):
  for n,h in build[group].items():assert sha(z.read('Garnet/'+n))==h
 policy_path='Garnet/test2026/cuda/qwen38_5090/evidence/b4-joined-mlp-source-policy.json';policy=doc(policy_path)
 install=doc('results/b4-joined-mlp-install-receipt.json')
 assert sha(z.read(policy_path))==workflow['source_policy_sha256']==install['source_policy_sha256']
 assert sha(z.read('results/b4-joined-mlp-install-receipt.json'))==workflow['install_receipt_sha256']
 assert install['status']=='experimental_b4_joined_admission_installed_unqualified' and install['native_library_sha256']==manifest['library_sha256']
 assert sha(z.read('Garnet/'+policy['file']))==policy['new_sha256']==install['new_sha256']
 assert sha(z.read('results/b4-joined-mlp-original.py'))==policy['old_sha256']==install['old_sha256']
 assert z.read('Garnet/'+policy['file']).decode()==policy['new_source'] and z.read('results/b4-joined-mlp-original.py').decode()==policy['old_source']
 assert policy['allowed_profile']=={'batch':4,'capacity':4096,'chunk':4,'shared_compressed_weights':True} and not policy['default_enabled']
 assert policy['decision_trace_archive_sha256']==manifest['linked_archives']['b4-mtp-state-wall-trace-attempt1-evidence.zip']
 prior='shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1';proof=doc('results/'+prior+'-verified.json')
 assert proof['model_gate_passed'] and proof['strict_same_profile_target_state_hash_gate_passed'] and proof['library_sha256']==manifest['library_sha256']
 assert proof['archive_sha256']==manifest['linked_archives'][prior+'-evidence.zip']
 checks=[];hidden_checks=[]
 report_path=base+'native/report.json'
 if report_path in manifest['files']:
  report=doc(report_path);assert report['batch']==4 and report['capacity']==4096 and report['chunk']==report['draft_chunk']==4
  assert report['joined_mlp_projections'] and report['draft_builder_workspace_bytes']==256<<20
  with zipfile.ZipFile(out/(prior+'-evidence.zip')) as old:
   oldbase='results/'+prior+'/native/';oldraw=old.read(oldbase+'report.json');reference=json.loads(oldraw)
   assert sha(oldraw)==report['retained_unjoined_b4_report_sha256']
   old_cases={(c['group'],c['repeat']):c for c in reference['cases']}
   for c in report['cases']:
    key=(c['group'],c['repeat']);ref=old_cases[key]
    exact=all(c[k]==ref[k] for k in ('generated_ids','expected_ids','ordinary_ids','ordinary_state','mtp_state','limits','prompt_lengths'))
    same=c['ordinary_state']==c['mtp_state'];ids=c['generated_ids']==c['ordinary_ids']==c['expected_ids']
    assert c['unjoined_b4_ids_and_state_exact']==(c['generated_ids']==ref['generated_ids'] and c['ordinary_state']==ref['ordinary_state'] and c['mtp_state']==ref['mtp_state'])
    checks.append({'group':key[0],'repeat':key[1],'retained_unjoined_exact':exact,'same_profile_state_exact':same,'ids_exact':ids})
    for mode in ('ordinary','mtp'):
     file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32'
     actual=z.read(base+'native/'+file);equal=actual==old.read(oldbase+file);hidden_checks.append(equal)
     assert sha(actual)==c[mode+'_state']['last_hidden_sha256']
    if passed:
     assert exact and same and ids and c['duplicate_pair_isolation_exact']
     m=c['decoder_memory'];assert m['packed_profiles']['joined_mlp_projections'] and m['packed_profiles']['host_embedding_lookup'] and m['packed_profiles']['deferred_state_commit']
     assert m['compressed_projection_pool_entries']==336
     bodies=[g for g in m['graphs'] if g['native_nvfp4_projection_weights']==128 and g['native_fp8_projection_weights']==208];assert len(bodies)==1
     assert bodies[0]['external_compressed_weight_inputs']==336 and bodies[0]['external_bf16_weight_inputs']==0 and bodies[0]['graph_capture_failures']==0
   if passed:assert report['status']=='backend_smoke_completed' and len(checks)==4 and len(hidden_checks)==8 and all(hidden_checks)
 else:assert not passed
 events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
 assert events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
 monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
 assert monitor['child_exit_code']==workflow['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
 peak=max(r['used_bytes'] for r in samples);assert peak==monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
result={'status':'b4_host_joined_mtp_correctness_replayed' if passed else 'b4_host_joined_mtp_failure_preserved','model_gate_passed':passed,'archive_sha256':sha(archive.read_bytes()),'library_sha256':manifest['library_sha256'],'files_verified':len(manifest['files']),'cases':checks,'raw_last_hidden_arrays_compared':len(hidden_checks),'raw_last_hidden_all_exact':bool(hidden_checks) and all(hidden_checks),'sampled_peak_adapter_used_bytes':peak,'performance_admitted':False,'scope':'B4/T4 host/deferred joined gate-up MTP3,4096capacity,256MiB draft builder. Exact retained unjoined B4 ID/state witnesses plus raw last-hidden bytes. No raw full-state arithmetic, stream/HTTP/broad-quality or speed admission.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
