"""Replay independent full-cache reference controls and raw FP64 witnesses."""
from pathlib import Path
import hashlib,json,platform,sys,zipfile
import numpy as np
from bf16_kv_overlay_oracle import ATOL,RTOL,compare_witness,sample_triples
out=Path(sys.argv[1]);name='bf16-kv-float-reference-native-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
plan=[dict(batch=4,chunk=t,layer=l,end=e) for t in (4,32) for l in (3,63) for e in (0,1,511,512,513,4095,4096)]
plan += [dict(batch=4,chunk=4,layer=l,end=513) for l in range(7,63,4)]
plan=[dict(c,id=i) for i,c in enumerate(plan)];numerical=[]
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');raw=z.read(methods+'evidence/bf16-kv-float-reference-native-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and w['library_unchanged'] and w['cleanup_finished'] and w['source_sha256']==w['source_after_sha256']==p['files']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 parent=doc('results/bf16-prefill-diagnostic-attempt1-verified.json');overlay=doc('results/bf16-kv-overlay-native-attempt1-verified.json')
 assert parent['status']=='bf16_prefill_diagnostic_replayed' and parent['library_sha256']==w['library_sha256']==p['observed_unused_library_sha256']==m['observed_unused_library_sha256']
 assert parent['archive_sha256']==w['parent_diagnostic_archive_sha256']==m['parent_diagnostic_archive_sha256']==sha((out/'bf16-prefill-diagnostic-attempt1-evidence.zip').read_bytes())
 assert overlay['status']=='bf16_kv_overlay_native_fp64_replayed' and len(overlay['cases'])==70 and overlay['maximum_tolerance_fraction']<=1
 assert overlay['archive_sha256']==m['qualified_overlay_archive_sha256']==p['qualified_overlay_archive_sha256']==sha((out/'bf16-kv-overlay-native-attempt1-evidence.zip').read_bytes())
 oracle=Path(__file__).with_name('bf16_kv_overlay_oracle.py');assert sha(oracle.read_bytes())==sha(z.read(methods+oracle.name))==overlay['oracle_sha256']
 receipt=doc(methods+'evidence/bf16-kv-float-reference-probe-local-compile.json')
 for n,h in receipt['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 assert sha(z.read('tools/bf16-kv-native/probe_bf16_kv_float_reference.exe'))==receipt['executable_sha256']
 assert sha(z.read(methods+'evidence/bf16-kv-float-reference-probe-compile.log'))==receipt['compile_log_sha256']
 assert sha(z.read(methods+'evidence/compile_bf16_kv_float_reference_probe.bat.txt'))==receipt['recipe_sha256']
 recipe=doc(methods+'evidence/bf16-kv-float-reference-probe-derivation.json');source=z.read(methods+recipe['source']);assert sha(source)==recipe['source_sha256']
 text=source.decode().replace('\r\n','\n')
 for before,after in recipe['changes']:assert text.count(before)==1;text=text.replace(before,after)
 assert text.encode()==z.read(methods+'probe_bf16_kv_float_reference.cu') and sha(text.encode())==recipe['result_sha256']
 passed=w['exit_code']==0;negative=graphs=roundtrips=overlay_negative=0;runtime=None
 if passed:
  assert sha(z.read(base+'regenerated-layout.h'))==p['fixture_header_sha256']==receipt['fixture_header_sha256']
  for ledger in ('cpu-oracle','cpu-fixture','native'):
   events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==0
  cpu=z.read(base+'cpu-oracle/commands.log').replace(b'\r\n',b'\n');assert b'Ran 8 tests' in cpu and b'\nOK' in cpu
  records=[]
  for line in z.read(base+'native/commands.log').splitlines():
   try:r=json.loads(line)
   except (ValueError,UnicodeDecodeError):continue
   if isinstance(r,dict):records.append(r)
  identities=[r for r in records if r.get('event')=='gpu_identity'];assert len(identities)==1 and identities[0]['uuid']==p['gpu_uuid'] and identities[0]['sm']==120 and '5090' in identities[0]['name']
  versions=[r for r in records if r.get('event')=='reference_runtime'];assert len(versions)==1;runtime=versions[0];assert runtime['cuda_runtime_version']>=13000 and runtime['cuda_driver_version']>=runtime['cuda_runtime_version']
  cases=[r for r in records if r.get('event')=='case_collected'];refs=[r for r in records if r.get('event')=='reference_case_verified'];rounds=[r for r in records if r.get('event')=='reference_roundtrip_verified']
  assert len(cases)==len(refs)==len(plan)==42 and [r['id'] for r in rounds]==[c['id'] for c in plan if c['end']]
  for r in rounds:assert r['context_bit_exact'] and r['full_cache_exact'] and r['readonly_previous_output_exact'] and r['canaries_exact']
  raw=z.read(base+'witnesses.bin');cursor=0
  for c,r,expected in zip(cases,refs,plan):
   for key,value in expected.items():assert c[key]==value
   assert r['id']==expected['id'] and r['context_bit_exact'] and r['dense_readonly_exact'] and r['canaries_exact'] and r['full_cache_words_checked']==33554432
   assert r['rejected_calls']==(108 if c['id'] in (0,14) else 0) and r['graph_replay_exact']==(c['id'] in (1,15))
   assert c['readonly_exact'] and c['canaries_exact'] and c['graph_replay_exact']==(c['id'] in (1,15)) and c['prefix_cache_exact']==bool(c['end'])
   assert c['rejected_calls']==(50 if c['id'] in (0,14) else 0) and c['samples']==[list(t) for t in sample_triples(expected)] and c['witness_offset']==cursor
   size=len(c['samples'])*256*4;assert c['witness_bytes']==size
   numerical.append(dict(expected,**compare_witness(expected,raw[cursor:cursor+size])));cursor+=size
   negative+=r['rejected_calls'];graphs+=r['graph_replay_exact'];overlay_negative+=c['rejected_calls']
  roundtrips=len(rounds);assert cursor==len(raw) and negative==216 and graphs==2 and roundtrips==38 and overlay_negative==100
  finals=[r for r in records if r.get('status')=='bf16_float_reference_collected_needs_fp64_replay'];assert len(finals)==1 and finals[0]['cases']==42 and finals[0]['witness_bytes']==len(raw) and not finals[0]['model_qualified'] and not finals[0]['performance_admitted']
  assert w['status']=='bf16_float_reference_collected_needs_independent_fp64_replay'
 resource=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples)
 assert resource['child_exit_code']==w['exit_code'] and resource['gpu']['uuid']==p['gpu_uuid'] and resource['sampled_memory']['samples']==len(samples) and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
 archived=sha(z.read(methods+'verify_bf16_kv_float_reference_native.py'))
result={'status':'bf16_float_reference_native_fp64_replayed' if passed else 'bf16_float_reference_native_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
 'cases':numerical,'reference_rejected_calls':negative,'overlay_rejected_calls':overlay_negative,'reference_graph_replays':graphs,'reference_roundtrips':roundtrips,
 'fp64_witness_scalars':sum(c['scalars'] for c in numerical),'maximum_absolute_error':max((c['max_absolute_error'] for c in numerical),default=None),'maximum_tolerance_fraction':max((c['max_tolerance_fraction'] for c in numerical),default=None),
 'absolute_tolerance':ATOL,'relative_tolerance':RTOL,'sampled_correctness_adapter_peak_bytes':peak,'runtime':runtime,'numpy_version':np.__version__,'python_version':platform.python_version(),
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'oracle_sha256':sha(oracle.read_bytes()),'model_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
