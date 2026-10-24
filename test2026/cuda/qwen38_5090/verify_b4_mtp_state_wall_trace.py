"""Replay closed diagnostic against retained B4 IDs/state and last-hidden bytes."""
from pathlib import Path
import collections,hashlib,json,math,sys,zipfile
out=Path(sys.argv[1]);name='b4-mtp-state-wall-trace-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;manifest=json.loads(z.read('evidence-manifest.json'))
 assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 for n,h in manifest['linked_archives'].items():assert sha((out/n).read_bytes())==h
 doc=lambda n:json.loads(z.read(n));base='results/'+name+'/'
 workflow=doc(base+'workflow.json');passed=workflow['exit_code']==0
 assert workflow['library_sha256']==manifest['library_sha256']
 build=doc('results/native-host-embedding-build-attempt1/manifest.json');assert build['exit_code']==0 and build['library_sha256']==manifest['library_sha256']
 for group in ('candidate_source_sha256','unchanged_policy_source_sha256'):
  for path,want in build[group].items():assert sha(z.read('Garnet/'+path))==want
 for n,h in workflow['workspace_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 for key,file in [('probe_sha256','probe_b4_mtp_state_wall_trace.py'),('full_correctness_probe_sha256','probe_shared_mtp3_full_b4_t4_host_embedding_workspace.py')]:assert sha(z.read('Garnet/test2026/cuda/qwen38_5090/'+file))==workflow[key]
 prior_name='shared-mtp3-full-b4-t4-host-embedding-workspace-attempt1'
 proof=doc('results/'+prior_name+'-verified.json');assert proof['model_gate_passed'] and proof['strict_same_profile_target_state_hash_gate_passed']
 assert proof['archive_sha256']==manifest['linked_archives'][prior_name+'-evidence.zip']
 assert proof['library_sha256']==manifest['library_sha256']
 http_name='batch4-t4-mtp-host-embedding-http-pilot-attempt1'
 http=doc('results/'+http_name+'-verified.json');assert http['status']=='batch4_t4_mtp_host_embedding_http_pilot_independently_verified'
 assert http['archive_sha256']==manifest['linked_archives'][http_name+'-evidence.zip']==workflow['prior_http_archive_sha256']
 report_path=base+'native/correctness/report.json';checks=[];raw_arrays=0
 if report_path in manifest['files']:
  report=doc(report_path)
  with zipfile.ZipFile(out/(prior_name+'-evidence.zip')) as old:
   oldbase='results/'+prior_name+'/native/';reference=json.loads(old.read(oldbase+'report.json'))
   by_case={(c['group'],c['repeat']):c for c in reference['cases']}
   for c in report['cases']:
    key=(c['group'],c['repeat']);ref=by_case[key]
    exact=all(c[k]==ref[k] for k in ('generated_ids','expected_ids','ordinary_ids','ordinary_state','mtp_state','limits','prompt_lengths'))
    assert c['generated_ids']==c['ordinary_ids']==c['expected_ids'] and c['ordinary_state']==c['mtp_state'] and exact
    checks.append({'group':key[0],'repeat':key[1],'exact_ids_state_and_retained_reference':exact})
    for mode in ('ordinary','mtp'):
     file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32'
     assert z.read(base+'native/correctness/'+file)==old.read(oldbase+file);raw_arrays+=1
  if passed:assert report['status']=='backend_smoke_completed' and len(checks)==4 and raw_arrays==8
 else:assert not passed
 stages=[];trace_path=base+'native/stages.jsonl'
 if trace_path in manifest['files']:
  stages=[json.loads(line) for line in z.read(trace_path).decode().splitlines()]
  previous=-math.inf
  for i,r in enumerate(stages):
   assert r['stage'] in ('target_chunk','target_program','prefix_commit','draft','draft_correct')
   assert all(math.isfinite(r[k]) for k in ('started','ended','wall_seconds')) and r['ended']>=r['started']
   assert abs(r['wall_seconds']-(r['ended']-r['started']))<1e-8 and r['ended']>=previous;previous=r['ended']
   assert type(r['depth']) is int and r['depth']>=0
   if passed:assert r['error'] is None
   if r['depth']:
    parent=next(v for v in stages[i+1:] if v['depth']==r['depth']-1)
    assert parent['stage']==r['parent'] and parent['started']<=r['started']<=r['ended']<=parent['ended']
   else:assert r['parent'] is None
 if passed:
  meta=doc(base+'native/trace-metadata.json');assert meta['status']=='correctness_finished_needs_independent_replay' and meta['source_sha256']==workflow['probe_sha256']
  assert stages and {s['stage'] for s in stages}=={'target_chunk','target_program','prefix_commit','draft','draft_correct'}
 events=[json.loads(line) for line in z.read(base+'ledger/commands.jsonl').decode().splitlines()]
 assert events[-1]['event']=='finished' and events[-1]['exit_code']==workflow['exit_code']
 monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
 assert monitor['child_exit_code']==workflow['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
 peak=max(r['used_bytes'] for r in samples);assert peak==monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
 counts=collections.Counter(s['stage'] for s in stages)
result={'status':'b4_mtp_state_wall_trace_replayed' if passed else 'b4_mtp_state_wall_trace_failure_preserved','correctness_passed':passed,'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'cases':checks,'raw_last_hidden_arrays_verified':raw_arrays,'stage_counts':dict(counts),'depth_zero_wall_seconds':{k:sum(s['wall_seconds'] for s in stages if s['stage']==k and s['depth']==0) for k in counts},'sampled_peak_adapter_used_bytes':peak,'performance_admitted':False,'scope':'Aggregate synchronized wall diagnostics across mixed long/short correctness cases. Parent includes child; never sum nested intervals. No per-case trace labels, isolated kernel timing, uninstrumented serving speed or independent full-state arithmetic.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
