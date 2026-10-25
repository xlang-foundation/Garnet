"""Replay native build lineage and CPU checks; no model/speed qualification."""
from pathlib import Path
import hashlib,json,re,sys,zipfile

out=Path(sys.argv[1]);name='integrated-batch-top1-native-build-attempt1';archive=out/(name+'-evidence.zip')
base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');raw=z.read(methods+'evidence/integrated-batch-top1-native-build-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 assert w['environment_files']==w['environment_files_after']==p['environment_files'] and w['cleanup_finished'] and not w['runtime_loaded']
 assert w['runtime_dependencies_before']==p['runtime_dependencies']
 assert w['retained_original_library_sha256']==p['runtime_dependencies']['garnet.dll']
 for n,h in p['runtime_dependencies'].items():
  if n!='garnet.dll':assert w['runtime_dependencies_after'][n]==h
 parent='integrated-batch-top1-correctness-attempt1';proofraw=z.read('results/'+parent+'-verified.json');proof=json.loads(proofraw)
 assert sha(proofraw)==p['component_proof_sha256']==w['component_proof_sha256']
 assert sha((out/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256']==p['component_archive_sha256']==w['component_archive_sha256']==manifest['component_archive_sha256']
 assert proof['status']=='integrated_batch_top1_correctness_replayed' and len(proof['variants'])==4 and not proof['native_runtime_built'] and not proof['performance_admitted']
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as previous:
  parent_w=json.loads(previous.read('results/'+parent+'/workflow.json'))
  assert parent_w['exit_code']==0 and parent_w['cleanup_finished'] and parent_w['library_unchanged']
  assert parent_w['source_sha256']==parent_w['source_after_sha256']==p['component_sources']
  assert all(p['files'][n]==h for n,h in p['component_sources'].items())
 observed=doc(methods+'evidence/integrated-top1-native-build-environment.json')
 assert observed['files']==p['environment_files'] and observed['runtime_dependencies']==p['runtime_dependencies']
 assert observed['observer_sha256']==sha(z.read(methods+'evidence/integrated-top1-native-build-environment-observer.py.txt'))
 assert sha(z.read(base+'CMakeCache-before.txt'))==p['cmake_cache_before_sha256']==observed['cmake_cache_sha256']
 passed=w['exit_code']==0
 if passed:
  assert w['status']=='integrated_batch_top1_native_built_model_unqualified'
  assert w['library_sha256']==w['runtime_dependencies_after']['garnet.dll']!=p['runtime_dependencies']['garnet.dll']
  assert sha(z.read('build/bin/garnet_batch_top1_policy_test.exe'))==w['policy_executable_sha256']==w['policy_executable_after_sha256']
  for ledger in ('cpu-ledger','build-ledger','policy-ledger'):
   records=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
   assert len(records)==2 and records[0]['event']=='started' and records[-1]['event']=='finished' and records[-1]['exit_code']==0 and records[0]['id']==records[-1]['id']
   log=z.read(base+ledger+'/commands.log');start,end=records[-1]['raw_start_offset'],records[-1]['raw_end_offset'];assert 0<=start<end<=len(log);log=log[start:end]
   if ledger=='cpu-ledger':assert records[0]['argv'][3:]==p['test_ids'] and int(re.search(rb'Ran (\d+) tests',log)[1])==len(p['test_ids']) and b'\nOK' in log
   if ledger=='policy-ledger':assert b'BATCH_TOP1_POLICY_PASS parse=9 boundaries=112' in log
   if ledger=='build-ledger':
    assert b'Linking CXX shared library' in log and b'garnet.dll' in log and b'cmake version' in log
    imports=log.lower().split(b'dump of file')[-1];assert b'dump of file' in log.lower()
    assert not any(n in imports for n in (b'cudart',b'nvcuda',b'cublas',b'nvinfer',b'garnet.dll'))
   for artifact,h in records[0]['artifact_sha256'].items():
    if artifact in p['environment_files']:assert h==p['environment_files'][artifact];continue
    path=Path(artifact);assert path.parts[1:3]==('CantorAI','5090-study-20261023');relative=Path(*path.parts[3:]).as_posix()
    if relative=='build/bin/garnet.dll':assert h==p['runtime_dependencies']['garnet.dll']
    else:assert sha(z.read(relative))==h
  cache=z.read('build/CMakeCache.txt')
  for token in (b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF',b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF',b'CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120'):assert token in cache
  ninja=z.read('build/build.ninja').decode().replace('\r\n','\n').replace('\\','/')
  for unit in ('cuda_tensor_io.cpp','logits_sampler_kernel.cu','logits_top1_batch_policy_test.cpp'):
   blocks=[b for b in ninja.split('\n\n') if b.startswith('build ') and '/'+unit+'.obj:' in b];assert len(blocks)==1 and ' -MT ' in blocks[0] and '/MD ' not in blocks[0] and '-MD ' not in blocks[0]
 else:assert w['status']=='integrated_batch_top1_native_build_failed_library_restored' and w['runtime_dependencies_after']==p['runtime_dependencies']
 archived=sha(z.read(methods+'verify_integrated_batch_top1_native_build.py'))
report={'status':'integrated_batch_top1_native_build_replayed' if passed else 'integrated_batch_top1_native_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),
 'files_verified':len(manifest['files']),'library_sha256':w.get('library_sha256'),'retained_original_library_sha256':w['retained_original_library_sha256'],
 'cpu_tests':len(p['test_ids']) if passed else None,'policy_settings':9 if passed else None,'policy_boundaries':112 if passed else None,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'runtime_dependencies_preserved':5,
 'gpu_model_executed':False,'model_quality_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 1)
