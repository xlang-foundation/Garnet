"""Replay both full-model modes against immutable prior lifecycle observations."""
from pathlib import Path
import hashlib,json,sys,zipfile
from integrated_top1_model_replay import compare

out=Path(sys.argv[1]);name='integrated-batch-top1-model-lifecycle-attempt1';archive=out/(name+'-evidence.zip')
base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest();variants=[];failed=[];peaks={}
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');raw=z.read(methods+'evidence/integrated-top1-model-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 assert w['library_unchanged'] and w['cleanup_finished'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and w['library_sha256']==p['runtime_dependencies']['garnet.dll']
 assert sha(Path(__file__).with_name('integrated_top1_model_replay.py').read_bytes())==p['files'][methods+'integrated_top1_model_replay.py']
 assert manifest['linked_archives']=={parent+'-evidence.zip':entry['archive_sha256'] for parent,entry in p['parents'].items()}
 for parent,entry in p['parents'].items():
  proofraw=z.read('results/'+parent+'-verified.json');proof=json.loads(proofraw)
  assert sha(proofraw)==entry['proof_sha256'] and proof['status']==entry['status'] and proof['archive_sha256']==entry['archive_sha256']==sha((out/(parent+'-evidence.zip')).read_bytes())
 parent='bf16-padded-lifecycle-scope-fixed-attempt2';prefix='results/'+parent+'/native/'
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as old_zip:
  old=json.loads(old_zip.read(prefix+'report.json'));old_vectors={n.removeprefix(prefix):old_zip.read(n) for n in old_zip.namelist() if n.startswith(prefix) and n.endswith('.f32')}
  recipe=doc(methods+'evidence/integrated-top1-model-probe-derivation.json');source=old_zip.read(methods+recipe['source']);assert sha(source)==recipe['source_sha256'];text=source.decode().replace('\r\n','\n')
  for before,after in recipe['changes']:assert text.count(before)==1;text=text.replace(before,after,1)
  assert text.encode()==z.read(methods+'probe_integrated_batch_top1_model.py') and sha(text.encode())==recipe['result_sha256']
 passed=w['exit_code']==0
 if passed:assert w['status']=='integrated_top1_full_model_lifecycle_finished_needs_replay' and [(v['label'],v['mode']) for v in w['variants']]==[('disabled','0'),('enabled','1')]
 for ledger in ('checker-ledger','cpu-ledger')+tuple(v['label']+'-ledger' for v in w['variants']):
  record_path=base+ledger+'/commands.jsonl'
  if record_path not in manifest['files']:assert not passed;continue
  records=[json.loads(line) for line in z.read(record_path).splitlines()];assert len(records)==2 and records[0]['event']=='started' and records[-1]['event']=='finished' and records[0]['id']==records[-1]['id']
  if passed:assert records[-1]['exit_code']==0
  if ledger=='checker-ledger' and records[-1]['exit_code']==0:assert b'Ran 7 tests' in z.read(base+ledger+'/commands.log') and b'\nOK' in z.read(base+ledger+'/commands.log')
  if ledger=='cpu-ledger':assert records[-1]['exit_code']==0 and b'Ran 11 tests' in z.read(base+ledger+'/commands.log') and b'\nOK' in z.read(base+ledger+'/commands.log')
  for artifact,h in records[0]['artifact_sha256'].items():
   path=Path(artifact);assert path.parts[1:3]==('CantorAI','5090-study-20261023');relative=Path(*path.parts[3:]).as_posix()
   if relative=='build/bin/garnet.dll':assert h==w['library_sha256']
   else:assert sha(z.read(relative))==h
 for entry in w['variants']:
  label,mode=entry['label'],entry['mode'];prefix=base+label+'/'
  if passed:
   assert entry['finished'] and entry['exit_code']==0 and entry['parent_parity_passed']
   r=doc(prefix+'report.json');vectors={n.removeprefix(prefix):z.read(n) for n in manifest['files'] if n.startswith(prefix) and n.endswith('.f32')}
   parity=compare(old,r,old_vectors,vectors,w['library_sha256'],mode);assert parity==doc(base+label+'-parent-parity.json');variants.append({'label':label,**parity})
  else:failed.append({'label':label,'available_report':prefix+'report.json' in manifest['files'],'raw_vectors':sum(n.startswith(prefix) and n.endswith('.f32') for n in manifest['files'])})
  monitor_path=base+label+'-resources/resource-monitor.json'
  if monitor_path in manifest['files']:
   monitor=doc(monitor_path);samples=[json.loads(line) for line in z.read(base+label+'-resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples);peaks[label]=peak
   assert monitor['child_exit_code']==entry['exit_code'] and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
   assert monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  else:assert not passed
 archived=sha(z.read(methods+'verify_integrated_batch_top1_model.py'))
report={'status':'integrated_top1_full_model_lifecycle_replayed' if passed else 'integrated_top1_full_model_lifecycle_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),
 'library_sha256':w['library_sha256'],'variants':variants,'failed_variants_preserved':failed,'correctness_sampled_peak_adapter_bytes':peaks,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'broad_quality_qualified':False,'actual_4k_qualified':False,'b8_qualified':False,'streaming_http_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 1)
