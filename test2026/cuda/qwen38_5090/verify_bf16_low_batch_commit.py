"""Replay immutable B1 commit observations against integer and frozen bytes."""
from pathlib import Path
import hashlib,json,re,sys,zipfile
from bf16_low_batch_commit_oracle import case_plan,compare,expected,samples
from replay_bf16_low_batch_commit_observations import replay

out=Path(sys.argv[1]);name='bf16-low-batch-commit-native-attempt1';archive=out/(name+'-evidence.zip')
base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest();checked=[];failures=[];peak=None
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');raw=z.read(methods+'evidence/bf16-low-batch-commit-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files']
 assert w['inherited_model_sources']==w['inherited_model_sources_after']==p['inherited_model_sources']
 assert w['tools']==w['tools_after']==p['tools'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
 assert w['library_unchanged'] and w['cleanup_finished'] and not w['runtime_loaded'] and w['observed_unused_library_sha256']==p['runtime_dependencies']['garnet.dll']
 assert w['parents']==manifest['parents']==p['parents']
 for n,h in {**p['inherited_model_sources'],**p['files']}.items():assert sha(z.read(n))==h
 for n in ('bf16_low_batch_commit_oracle.py','bf16_low_batch_layout.py','replay_bf16_low_batch_commit_observations.py'):assert sha(Path(__file__).with_name(n).read_bytes())==p['files'][methods+n]
 for parent,entry in p['parents'].items():
  raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw)
  assert sha(raw)==entry['proof_sha256'] and proof['status']==entry['status'] and sha((out/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256']==entry['archive_sha256']
 parent='integrated-batch-top1-model-lifecycle-attempt1';model=doc('results/'+parent+'-verified.json')
 assert model['status'] in ('integrated_top1_full_model_lifecycle_replayed','integrated_top1_full_model_lifecycle_failure_preserved') and not model['performance_admitted']
 assert model['library_sha256']==w['observed_unused_library_sha256']
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as previous:
  prior=json.loads(previous.read('results/'+parent+'/workflow.json'))
  assert prior['cleanup_finished'] and prior['library_unchanged'] and prior['source_sha256']==prior['source_after_sha256']==p['inherited_model_sources']
  assert prior['runtime_dependencies']==prior['runtime_dependencies_after']==p['runtime_dependencies']
  assert (prior['exit_code']==0)==(model['status']=='integrated_top1_full_model_lifecycle_replayed')
 # Reconstruct candidate sources and the retained all-byte checker prefix.
 recipe=doc(methods+'evidence/bf16-low-batch-commit-source-derivation.json')
 for target,entry in recipe['files'].items():
  raw=z.read('Garnet/'+entry['source']);assert sha(raw)==entry['source_sha256'];text=raw.decode().replace('\r\n','\n')
  for before,after,count in entry['changes']:assert text.count(before)==count;text=text.replace(before,after)
  assert text.encode()==z.read(methods+target) and sha(text.encode())==entry['result_sha256']
 entry=doc(methods+'evidence/bf16-low-batch-commit-probe-derivation.json');raw=z.read('Garnet/'+entry['source']);assert sha(raw)==entry['source_sha256'];text=raw.decode().replace('\r\n','\n');text=text[:text.index(entry['prefix_before_marker'])]
 for before,after,count in entry['prefix_changes']:assert text.count(before)==count;text=text.replace(before,after)
 assert (text+entry['new_body']).encode()==z.read(methods+'probe_bf16_low_batch_commit.cu') and sha((text+entry['new_body']).encode())==entry['result_sha256']
 # Actual pre-GPU expected bytes are archived independently of GPU collection.
 frozen=doc(methods+'evidence/bf16-low-batch-commit-byte-reference.json');frozen_raw=z.read('results/bf16-low-batch-commit-expected.bin')
 assert not frozen['gpu_executed'] and frozen['model_outputs']==0 and frozen['tolerance']=='Exact byte equality; no floating tolerance'
 assert sha(frozen_raw)==frozen['sha256'] and len(frozen_raw)==frozen['bytes']==104704
 plan=case_plan();assert [r['case'] for r in frozen['cases']]==plan and len(plan)==20
 cursor=0
 for case,row in zip(plan,frozen['cases']):
  wanted=expected(case);assert row['offset']==cursor and row['bytes']==len(wanted) and row['samples']==len(samples(case)) and row['sha256']==sha(wanted)
  assert frozen_raw[cursor:cursor+len(wanted)]==wanted;cursor+=len(wanted)
 assert cursor==len(frozen_raw)
 for n,h in frozen['oracle_sha256'].items():assert sha(z.read(methods+n))==h
 assert sha(z.read('tools/bf16-kv-native/bf16_kv_fixture_layout.h'))==p['fixture_header_sha256']==frozen['fixture_regenerated_from_b1_layout_sha256']
 owner=doc('tools/bf16-kv-native/low-batch-commit-owner.json');assert Path(owner['study'])==Path('D:/CantorAI/5090-study-20261023') and owner['component']=='bf16-low-batch-commit'
 for key,file in [('executable','bf16-low-batch-commit.exe'),('policy_executable','low-batch-commit-policy.exe')]:
  if key+'_sha256' in w:assert sha(z.read('tools/bf16-kv-native/'+file))==w[key+'_sha256']==w[key+'_after_sha256']
 if 'executable_sha256' in w:assert Path(z.read('tools/bf16-kv-native/low-batch-commit-host-compiler.txt').decode().splitlines()[0])==Path(p['host_compiler'])
 collected=w['exit_code']==0
 for ledger in ('cpu-ledger','build-ledger','gpu-ledger'):
  path=base+ledger+'/commands.jsonl'
  if path not in manifest['files']:assert not collected;continue
  events=[json.loads(line) for line in z.read(path).splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[0]['id']==events[-1]['id']
  if collected:assert events[-1]['exit_code']==0
  for artifact,h in events[0]['artifact_sha256'].items():
   if artifact in p['tools']:assert h==p['tools'][artifact];continue
   a=Path(artifact);assert a.parts[1:3]==('CantorAI','5090-study-20261023');relative=Path(*a.parts[3:]).as_posix();assert sha(z.read(relative))==h
 if collected:
  assert w['status']=='bf16_low_batch_commit_collected_needs_byte_replay'
  cpu_log=z.read(base+'cpu-ledger/commands.log');cpu_events=[json.loads(line) for line in z.read(base+'cpu-ledger/commands.jsonl').splitlines()]
  assert int(re.search(rb'Ran (\d+) tests',cpu_log)[1])==len(p['test_ids']) and cpu_events[0]['argv'][3:]==p['test_ids'] and b'\nOK' in cpu_log
  assert b'LOW_BATCH_COMMIT_CPU_POLICY_PASS cases=63' in z.read(base+'build-ledger/commands.log')
  imports=z.read('tools/bf16-kv-native/low-batch-commit-policy-imports.txt').lower();assert not any(n in imports for n in (b'cudart',b'nvcuda',b'cublas',b'nvinfer',b'garnet.dll'))
  records=[]
  for line in z.read(base+'gpu-ledger/commands.log').splitlines():
   if line.startswith(b'{'):
    item=json.loads(line)
    if 'event' in item or 'status' in item:records.append(item)
  try:checked=replay(records,z.read(base+'witnesses.bin'),frozen_raw)
  except ValueError as error:failures.append({'error':str(error)})
 path=base+'resources/resource-monitor.json'
 if path in manifest['files']:
  monitor=doc(path);resources=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(r['used_bytes'] for r in resources)
  assert monitor['child_exit_code']==w['exit_code'] and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
  assert monitor['sampled_memory']['samples']==len(resources) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
 else:assert not collected
 archived=sha(z.read(methods+'verify_bf16_low_batch_commit.py'))
passed=collected and not failures
report={'status':'bf16_low_batch_commit_exact_bytes_replayed' if passed else 'bf16_low_batch_commit_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
 'observed_unused_library_sha256':w['observed_unused_library_sha256'],'closed_parent_model_status':model['status'],'cases':checked,'byte_failures':failures,
 'frozen_reference_sha256':frozen['sha256'],'correctness_sampled_peak_adapter_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
 'native_helper_plugin_model_admitted':False,'model_quality_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 1)
