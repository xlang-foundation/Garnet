"""Replay source/build lineage; valid engine/model behavior remains unqualified."""
from pathlib import Path
import hashlib,io,json,re,sys,zipfile
out=Path(sys.argv[1]);name='bf16-float-reference-model-build-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
 w=doc(base+'manifest.json');raw=z.read(methods+'evidence/bf16-float-reference-model-build-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and p['parent_archives']==m['linked_archives']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 assert w['candidate_source_sha256']==p['candidate_source_sha256'] and w['original_library_sha256']==p['original_library_sha256']
 source=z.read('results/bf16-float-reference-model-source.zip');assert sha(source)==w['source_zip_sha256']==p['source_zip_sha256']
 with zipfile.ZipFile(io.BytesIO(source)) as s:
  assert s.testzip() is None and set(s.namelist())==set(p['candidate_source_sha256'])
  for n,h in p['candidate_source_sha256'].items():assert sha(s.read(n))==h
 with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as s:
  assert s.testzip() is None and set(s.namelist())==set(w['original_source_sha256'])
  for n,h in w['original_source_sha256'].items():assert sha(s.read(n))==h==p['base_source_sha256'][n]
 passed=w['status']=='bf16_float_reference_native_build_finished_unqualified' and w['exit_code']==0
 if passed:
  assert w['runtime_dependencies_before_sha256']==w['runtime_dependencies_after_sha256'] and len(w['runtime_dependencies_after_sha256'])==5
  assert w['library_sha256']!=w['original_library_sha256']
  for n,h in p['candidate_source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
  for ledger in ('cpu-ledger','build-ledger'):
   events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished' and events[1]['exit_code']==0 and events[0]['id']==events[1]['id']
   raw=z.read(base+ledger+'/commands.log');e=events[-1];assert 0<=e['raw_start_offset']<e['raw_end_offset']<=len(raw);log=raw[e['raw_start_offset']:e['raw_end_offset']]
   if ledger=='cpu-ledger':
    tests=int(re.search(rb'Ran (\d+) tests',log)[1]);assert tests==49 and p['expected_cpu_tests']==50 and b'\nOK' in log
   else:assert b'Linking CXX shared library' in log and b'garnet.dll' in log
  cache=z.read('build/CMakeCache.txt');ninja=z.read('build/build.ninja').decode().replace('\r\n','\n').replace('\\','/')
  for token in (b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF',b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF'):assert token in cache
  for unit in ('qwen38_bf16_kv_float_reference_plugin.cpp','trt_builder.cpp'):
   blocks=[b for b in ninja.split('\n\n') if b.startswith('build ') and '/'+unit+'.obj:' in b];assert len(blocks)==1 and ' -MT ' in blocks[0] and '/MD ' not in blocks[0] and '-MD ' not in blocks[0]
  assert '/qwen38_bf16_kv_float_reference.cu.obj:' in ninja
 else:assert w['status']=='bf16_float_reference_native_build_failed_restored' and w['exit_code']!=0
 archived=sha(z.read(methods+'verify_bf16_float_reference_model_build.py'))
if passed:
 completion=out/'bf16-float-reference-cpu-completion-attempt1-evidence.zip'
 with zipfile.ZipFile(completion) as c:
  d=lambda n:json.loads(c.read(n));cm=d('evidence-manifest.json');cb='results/bf16-float-reference-cpu-completion-attempt1/'
  assert c.testzip() is None and len(c.namelist())==len(set(c.namelist()))==len(cm['files'])+1
  for n,h in cm['files'].items():assert sha(c.read(n))==h
  cpraw=c.read(methods+'evidence/bf16-float-reference-cpu-completion-preparation.json');cp=json.loads(cpraw);cw=d(cb+'workflow.json')
  assert cw['status']=='bf16_float_reference_cpu_completion_finished_needs_replay' and cw['exit_code']==0 and cw['library_unchanged'] and cw['library_sha256']==w['library_sha256']
  assert sha(cpraw)==cw['preparation_sha256'] and cp['build_archive_sha256']==cw['build_archive_sha256']==sha(archive.read_bytes())
  assert cm['linked_archives']=={archive.name:sha(archive.read_bytes())} and cw['original_cpu_tests']==tests==49
  assert cw['test_source_after_sha256']==cp['test_source_sha256'] and cw['product_source_after_sha256']==cp['product_source_sha256']
  assert cp['product_source_sha256']=={n:h for n,h in {**p['base_source_sha256'],**p['candidate_source_sha256']}.items() if not n.startswith('test2026/')}
  assert cw['test_ids']==cp['test_ids'] and len(set(cw['test_ids']))==len(cw['test_ids'])==50
  for n,h in cp['test_source_sha256'].items():assert sha(c.read('Garnet/'+n))==h
  for n,h in cp['methods'].items():assert sha(c.read(methods+n))==h
  inventory=d('results/bf16-float-reference-cpu-inventory.json');assert len(inventory['test_ids'])==49
  assert set(cw['test_ids'])-set(inventory['test_ids'])=={'test2026.cuda.qwen38_5090.test_joined_mlp_profile.JoinedMLPProfileTests.test_bounded_batch4_admission'}
  assert not set(inventory['test_ids'])-set(cw['test_ids'])
  assert sha(c.read(cb+'original-test.py.txt'))==inventory['source_sha256'][cp['target']]
  with zipfile.ZipFile(io.BytesIO(c.read('results/bf16-float-reference-cpu-inventory-source.zip'))) as prior:
   for n,h in inventory['source_sha256'].items():assert sha(prior.read(n))==h
  changed=[n for n in cp['test_source_sha256'] if sha(c.read('Garnet/'+n).replace(b'\r\n',b'\n'))!=inventory['normalized_source_sha256'][n]]
  assert changed==[cp['target']]
  events=[json.loads(l) for l in c.read(cb+'cpu-ledger/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==0
  log=c.read(cb+'cpu-ledger/commands.log');e=events[-1];log=log[e['raw_start_offset']:e['raw_end_offset']];assert b'Ran 50 tests' in log and b'\nOK' in log
r={'status':'bf16_float_reference_model_build_replayed' if passed else 'bf16_float_reference_model_build_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
   'library_sha256':w.get('library_sha256'),'original_library_sha256':w['original_library_sha256'],'initial_cpu_tests':tests if passed else None,'cpu_tests':50 if passed else None,'cpu_completion_archive_sha256':sha(completion.read_bytes()) if passed else None,
   'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'gpu_enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
