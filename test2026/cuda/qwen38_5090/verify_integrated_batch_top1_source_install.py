"""Replay the source-only install archive without loading a native library."""
from pathlib import Path
import hashlib,json,sys,zipfile

out=Path(sys.argv[1]);name='integrated-batch-top1-source-install-attempt1';archive=out/(name+'-evidence.zip')
base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 record=doc(base+'manifest.json');passed=record['exit_code']==0
 prepraw=z.read(methods+'evidence/integrated-batch-top1-source-install-preparation.json');p=json.loads(prepraw)
 assert sha(prepraw)==record['preparation_sha256'] and p['base_source_sha256']==record['base_source_sha256']
 assert p['candidate_source_sha256']==record['candidate_source_sha256']
 assert record['runtime_dependencies_before']==record['runtime_dependencies_after']==p['runtime_dependencies']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 parent='bf16-padded-lifecycle-scope-fixed-attempt2';raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw)
 assert sha(raw)==p['parent_proof_sha256']==record['parent_proof_sha256']
 assert proof['status']=='bf16_padded_lifecycle_replayed' and len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102
 assert not proof['performance_admitted'] and proof['library_sha256']==p['runtime_dependencies']['garnet.dll']
 assert sha((out/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256']==p['parent_archive_sha256']==record['parent_archive_sha256']==manifest['parent_archive_sha256']
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as parent_zip:
  previous=json.loads(parent_zip.read('results/'+parent+'/workflow.json'))
  assert previous['exit_code']==0 and previous['cleanup_finished'] and previous['library_unchanged']
  assert previous['source_sha256']==previous['source_after_sha256']==p['qualified_parent_source_sha256']
  assert all(p['base_source_sha256'][n]==h for n,h in previous['source_sha256'].items())
 bundle=z.read('results/integrated-batch-top1-source.zip');assert sha(bundle)==p['source_zip_sha256']==record['source_zip_sha256']
 import io
 with zipfile.ZipFile(io.BytesIO(bundle)) as packet:
  assert packet.testzip() is None and set(packet.namelist())==set(p['candidate_source_sha256'])
  assert len(packet.namelist())==len(p['candidate_source_sha256'])==5
  for n,h in p['candidate_source_sha256'].items():assert sha(packet.read(n))==h
 old_names=set(p['candidate_source_sha256'])-set(p['allowed_new_source']);assert len(old_names)==3 and len(p['allowed_new_source'])==2
 with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as original:
  assert original.testzip() is None and len(original.namelist())==len(old_names) and set(original.namelist())==old_names
  for n in old_names:assert sha(original.read(n))==p['base_source_sha256'][n]
 if passed:
  assert record['status']=='sources_installed_native_library_unbuilt'
  assert record['final_source_sha256']==p['candidate_source_sha256']
 else:assert record['status']=='source_install_failed_restore_attempted'
 archived=sha(z.read(methods+'verify_integrated_batch_top1_source_install.py'))
result={'status':'integrated_batch_top1_sources_installed_replayed' if passed else 'integrated_batch_top1_source_install_failure_preserved',
 'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'preparation_sha256':sha(prepraw),
 'parent_archive_sha256':proof['archive_sha256'],'source_zip_sha256':p['source_zip_sha256'],
 'observed_unused_library_sha256':p['runtime_dependencies']['garnet.dll'],'candidate_source_sha256':p['candidate_source_sha256'],
 'original_sources_preserved':3,'new_sources':2,'native_runtime_built':False,'model_qualified':False,'performance_admitted':False,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
