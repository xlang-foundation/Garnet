"""Replay first-use layer records and strict retained joined B4 correctness."""
from pathlib import Path
import hashlib,json,math,sys,zipfile
out=Path(sys.argv[1]);name='b4-joined-layer-profile-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;m=json.loads(z.read('evidence-manifest.json'))
 assert len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
 doc=lambda n:json.loads(z.read(n));base='results/'+name+'/'
 w=doc(base+'workflow.json');passed=w['exit_code']==0;assert w['library_sha256']==m['library_sha256'] and w['joined_mlp_projections']
 for n,h in w['workspace_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 for field,n in [('probe_sha256','probe_b4_joined_layer_profile.py'),('preflight_source_sha256','probe_trt_layer_profiler.py'),('full_correctness_probe_sha256','probe_b4_host_joined_mtp.py')]:assert sha(z.read('Garnet/test2026/cuda/qwen38_5090/'+n))==w[field]
 build=doc('results/native-host-embedding-build-attempt1/manifest.json');assert build['exit_code']==0 and build['library_sha256']==m['library_sha256']
 for group in ('candidate_source_sha256','unchanged_policy_source_sha256'):
  for n,h in build[group].items():assert sha(z.read('Garnet/'+n))==h
 policy_path='Garnet/test2026/cuda/qwen38_5090/evidence/b4-joined-mlp-source-policy.json';policy=doc(policy_path)
 assert sha(z.read(policy_path))==w['joined_policy_sha256'] and sha(z.read('Garnet/'+policy['file']))==policy['new_sha256']
 parent='b4-joined-delay10-http-pilot-attempt1';priorw=doc('results/'+parent+'/workflow.json')
 assert w['prior_http_archive_sha256']==m['linked_archives'][parent+'-evidence.zip']
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as prior:assert prior.read('results/'+parent+'/workflow.json')==z.read('results/'+parent+'/workflow.json')
 checks=[];arrays=0;layers=[]
 reference='shared-mtp3-full-b4-t4-host-joined-attempt1';proof=doc('results/'+reference+'-verified.json')
 assert proof['model_gate_passed'] and proof['archive_sha256']==m['linked_archives'][reference+'-evidence.zip'] and proof['library_sha256']==m['library_sha256']
 report_path=base+'native/correctness/report.json'
 if report_path in m['files']:
  report=doc(report_path)
  with zipfile.ZipFile(out/(reference+'-evidence.zip')) as old:
   oldbase='results/'+reference+'/native/';oldcases={(c['group'],c['repeat']):c for c in json.loads(old.read(oldbase+'report.json'))['cases']}
   for c in report['cases']:
    key=(c['group'],c['repeat']);ref=oldcases[key]
    exact=all(c[k]==ref[k] for k in ('generated_ids','ordinary_ids','expected_ids','ordinary_state','mtp_state','speculative_metrics'))
    checks.append({'group':key[0],'repeat':key[1],'retained_reference_exact':exact})
    for mode in ('ordinary','mtp'):
     file='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-hidden.f32'
     same=z.read(base+'native/correctness/'+file)==old.read(oldbase+file)
     if passed:assert same
     arrays+=1
    if passed:assert exact and c['unjoined_b4_ids_and_state_exact'] and c['full_target_state_exact']
  if passed:assert report['status']=='backend_smoke_completed' and len(checks)==4 and arrays==8
 else:assert not passed
 def parse_layers(path):
  rows=[]
  for line in z.read(path).decode().splitlines():
   f=line.split('\t')
   if f==['engine','layer','milliseconds']:continue
   assert len(f)==3;ms=float(f[2]);assert math.isfinite(ms) and ms>=0
   rows.append({'engine':f[0],'layer':f[1],'milliseconds':ms})
  return rows
 if passed:
  preflight=doc(base+'native/preflight/report.json');assert preflight['status']=='passed' and preflight['startup_profile'] and preflight['library_sha256']==m['library_sha256']
  assert parse_layers(base+'native/preflight/layers.tsv')==preflight['layers']
  layer_report=doc(base+'native/layer-profile-report.json');assert layer_report['status']=='correctness_and_layer_capture_finished_needs_replay' and layer_report['library_sha256']==m['library_sha256']
  for n,h in layer_report['source_sha256'].items():assert sha(z.read('Garnet/test2026/cuda/qwen38_5090/'+n))==h
  layers=parse_layers(base+'native/layers.tsv');assert layers==layer_report['layers'] and layers
  assert sha(z.read(base+'native/layers.tsv'))==layer_report['profile_sha256']
 events=[json.loads(l) for l in z.read(base+'ledger/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
 monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()]
 assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples)
 peak=max(r['used_bytes'] for r in samples);assert peak==monitor['sampled_memory']['sampled_peak_adapter_used_bytes']
result={'status':'b4_joined_layer_profile_replayed' if passed else 'b4_joined_layer_profile_failure_preserved','correctness_passed':passed,'archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),'cases':checks,'raw_last_hidden_arrays_compared':arrays,'layer_rows':len(layers),'engine_ids':sorted({r['engine'] for r in layers}),'sampled_peak_adapter_used_bytes':peak,'performance_admitted':False,'scope':'Existing IProfiler first-use eager records only. Successful same-DLL normalization preflight is output-hash/repeat witness, not independent raw norm arithmetic. Full-model IDs/state/counters and raw last-hidden compared to qualified joined reference. No warm graph, production latency, device-active attribution or speedup.'}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
