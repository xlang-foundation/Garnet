"""Replay B1 source installation and native build provenance without loading DLLs."""
from pathlib import Path
import hashlib,json,sys,zipfile
sha=lambda b:hashlib.sha256(b).hexdigest()
out=Path(sys.argv[1]);name='bf16-low-batch-native-build-attempt1';archive=out/(name+'-evidence.zip');base='results/'+name+'/';method='Garnet/test2026/cuda/qwen38_5090/'
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None
 assert len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');p=doc(method+'evidence/bf16-low-batch-native-build-preparation.json')
 assert sha(z.read(method+'evidence/bf16-low-batch-native-build-preparation.json'))==w['preparation_sha256']
 assert w['parents']==p['parents'] and w['methods']==w['methods_after']==p['methods']
 assert w['environment_before']==w['environment_after']==p['environment_files'] and w['runtime_before']==p['runtime_dependencies']
 assert w['source_before']==p['base_files'] and w['candidate_files']==p['candidate_files'] and w['candidate_base']==p['candidate_base']
 assert len(p['candidate_files'])==20 and len(p['product_files'])==19 and sum(h is not None for h in p['candidate_base'].values())==9
 assert set(p['candidate_files'])==set(p['product_files'])|{p['shared_test_fixture']}
 assert w['cleanup_finished'] and not w['runtime_loaded'] and w['retained_library_sha256']==p['runtime_dependencies']['garnet.dll']
 for parent,row in p['parents'].items():
  raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw)
  assert sha(raw)==row['proof_sha256'] and proof['status']==row['status'] and proof['archive_sha256']==row['archive_sha256']
 for n,h in p['methods'].items():assert sha(z.read(n))==h
 with zipfile.ZipFile(__import__('io').BytesIO(z.read(base+'original-source.zip'))) as original:
  expected={n:h for n,h in p['candidate_base'].items() if h is not None};assert set(original.namelist())==set(expected)
  for n,h in expected.items():assert sha(original.read(n))==h
 assert sha(z.read(p['source_zip']))==p['source_zip_sha256']==w['source_zip_sha256']
 with zipfile.ZipFile(__import__('io').BytesIO(z.read(p['source_zip']))) as candidate:
  assert set(candidate.namelist())==set(p['candidate_files'])
  for n,h in p['candidate_files'].items():assert sha(candidate.read(n))==h
 passed=w['exit_code']==0
 if passed:
  assert w['status']=='bf16_low_batch_native_built_model_unqualified' and len(p['test_modules'])==49
  expected=dict(p['base_files']);expected.update({'Garnet/'+n:h for n,h in p['candidate_files'].items()})
  assert w['installed_source']==w['source_after']==expected
  for n,h in expected.items():assert sha(z.read(n))==h
  for n,h in p['runtime_dependencies'].items():
   if n!='garnet.dll':assert w['runtime_after'][n]==h
  assert w['runtime_after']['garnet.dll']==w['library_sha256']!=p['runtime_dependencies']['garnet.dll']
  assert b'Ran 49 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
  for ledger in ('cpu-ledger','build-ledger'):
   records=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished' and records[0]['id']==records[1]['id'] and records[1]['exit_code']==0
  ninja=z.read('build/build.ninja')
  for unit in ('cuda_bf16_low_batch_kv_delta_commit.cpp','qwen38_bf16_low_batch_kv_overlay_plugin.cpp','qwen38_bf16_kv_low_batch_overlay.cu','qwen38_bf16_low_batch_kv_commit.cu'):assert unit.encode() in ninja
 archived=sha(z.read(method+'verify_bf16_low_batch_native_build.py'))
report={'status':'bf16_low_batch_native_build_replayed' if passed else 'bf16_low_batch_native_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w.get('library_sha256'),'retained_library_sha256':w['retained_library_sha256'],'source_files_installed':19 if passed else 0,'cpu_tests_passed':49 if passed else None,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'runtime_loaded':False,'gpu_model_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));raise SystemExit(0 if passed else 1)
