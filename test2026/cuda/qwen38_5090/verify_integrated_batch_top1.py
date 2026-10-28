"""Verify archive identity and replay all four production API mode processes."""
from pathlib import Path
import hashlib,json,platform,sys,zipfile
import numpy as np
from replay_integrated_top1_observations import replay

out=Path(sys.argv[1]);name='integrated-batch-top1-correctness-attempt1';archive=out/(name+'-evidence.zip')
base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
replayed=[];peaks={};failed_raw=[]
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0
 p_raw=z.read(methods+'evidence/integrated-batch-top1-preparation.json');p=json.loads(p_raw)
 assert sha(p_raw)==w['preparation_sha256'] and p['files']==w['source_sha256']==w['source_after_sha256']
 assert p['tools']==w['tools']==w['tools_after'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
 assert w['library_unchanged'] and w['preserved_component_executable_unchanged'] and w['cleanup_finished'] and not w['runtime_loaded']
 assert w['observed_unused_library_sha256']==p['observed_unused_library_sha256']==p['runtime_dependencies']['garnet.dll']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 for helper in ('integrated_top1_oracle.py','replay_integrated_top1_observations.py'):
  assert sha(Path(__file__).with_name(helper).read_bytes())==p['files'][methods+helper]
 parent='bf16-padded-lifecycle-scope-fixed-attempt2';parent_raw=z.read('results/'+parent+'-verified.json');parent_proof=json.loads(parent_raw)
 assert sha(parent_raw)==p['parent_proof_sha256']==w['parent_proof_sha256']
 assert parent_proof['status']=='bf16_padded_lifecycle_replayed' and len(parent_proof['cases'])==14 and parent_proof['raw_hidden_and_normalized_arrays']==102 and not parent_proof['performance_admitted']
 assert sha((out/(parent+'-evidence.zip')).read_bytes())==parent_proof['archive_sha256']==p['parent_archive_sha256']==w['parent_archive_sha256']==manifest['parent_archive_sha256']
 assert parent_proof['library_sha256']==p['observed_unused_library_sha256']
 install='integrated-batch-top1-source-install-attempt1';install_raw=z.read('results/'+install+'-verified.json');installed=json.loads(install_raw)
 assert sha(install_raw)==w['source_install_proof_sha256'] and installed['status']=='integrated_batch_top1_sources_installed_replayed'
 assert installed['preparation_sha256']==p['source_install_preparation_sha256']
 assert installed['source_zip_sha256']==p['installed_source_zip_sha256'] and installed['candidate_source_sha256']==p['installed_candidate_source_sha256']
 assert installed['observed_unused_library_sha256']==p['observed_unused_library_sha256']
 assert not installed['native_runtime_built'] and not installed['model_qualified'] and not installed['performance_admitted']
 assert installed['replay_verifier_sha256']==installed['archived_replay_verifier_sha256']==sha(z.read(methods+'verify_integrated_batch_top1_source_install.py'))
 assert sha((out/(install+'-evidence.zip')).read_bytes())==installed['archive_sha256']==w['source_install_archive_sha256']==manifest['source_install_archive_sha256']
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as parent_zip:
  parent_w=json.loads(parent_zip.read('results/'+parent+'/workflow.json'))
  assert parent_w['source_sha256']==parent_w['source_after_sha256']==p['qualified_parent_source_sha256'] and parent_w['exit_code']==0 and parent_w['cleanup_finished']
 operator='parallel-batch-top1-correctness-attempt1';op_raw=z.read('results/'+operator+'-verified.json');op=json.loads(op_raw)
 assert sha(op_raw)==p['operator_proof_sha256']==w['operator_proof_sha256']
 assert op['status']=='parallel_batch_top1_correctness_replayed' and op['cases_checked']==330 and op['raw_observations_checked']==1000 and not op['performance_admitted']
 assert sha((out/(operator+'-evidence.zip')).read_bytes())==op['archive_sha256']==p['operator_archive_sha256']==w['operator_archive_sha256']==manifest['operator_archive_sha256']
 with zipfile.ZipFile(out/(operator+'-evidence.zip')) as old_zip:
  for old,new in [('Garnet/src/cuda/logits_sampler_kernel.cu',methods+'evidence/parallel-batch-top1-legacy-sampler.cu.txt'),
                  ('Garnet/src/cuda/logits_top1_batch_parallel_candidate.cuh','Garnet/src/cuda/logits_top1_batch_parallel_candidate.cuh'),
                  (methods+'probe_parallel_batch_top1.cu',methods+'probe_parallel_batch_top1.cu')]:
   assert z.read(new)==old_zip.read(old)
 fixture=doc(methods+'evidence/integrated-batch-top1-fixture-oracles.json')
 assert not fixture['gpu_run'] and fixture['original_fixture_sha256']==sha(z.read(methods+'evidence/parallel-top1-fixture-oracles.json'))
 for n,h in fixture['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 for recipe,target in [(fixture['oracle_derivation'],'integrated_top1_oracle.py'),(doc(methods+'evidence/integrated-batch-top1-probe-derivation.json'),'probe_integrated_batch_top1.cu')]:
  raw=z.read(methods+recipe['source']);assert sha(raw)==recipe['source_sha256'];text=raw.decode().replace('\r\n','\n')
  for item in recipe['replacements']:
   before,after=item[:2];count=item[2] if len(item)==3 else 1
   assert text.count(before)==count;text=text.replace(before,after)
  assert text.encode()==z.read(methods+target) and sha(text.encode())==recipe['result_sha256']
 owner=doc('tools/parallel-batch-top1/study-owner.json');assert Path(owner['study'])==Path('D:/CantorAI/5090-study-20261023') and owner['component']=='parallel-batch-top1'
 for key,file in [('executable','integrated-probe.exe'),('policy_executable','integrated-policy.exe')]:
  if key+'_sha256' in w:assert sha(z.read('tools/parallel-batch-top1/'+file))==w[key+'_sha256']==w[key+'_after_sha256']
 if 'executable_sha256' in w:
  assert Path(z.read('tools/parallel-batch-top1/integrated-host-compiler.txt').decode().splitlines()[0])==Path(p['host_compiler'])
 labels=[('disabled','0',False),('enabled','1',True),('unset',None,False),('malformed','01',False)]
 if passed:
  assert w['status']=='integrated_batch_top1_correctness_finished_needs_replay'
  assert w['variants']==[{'label':label,'environment_value':value,'expected_enabled':enabled,'finished':True,'exit_code':0} for label,value,enabled in labels]
 for ledger in ['cpu-ledger','build-ledger','policy-ledger']+[label+'-ledger' for label,_,_ in labels]:
  n=base+ledger+'/commands.jsonl'
  if n not in manifest['files']:assert not passed;continue
  events=[json.loads(line) for line in z.read(n).splitlines()]
  assert len(events)==2 and events[0]['event']=='started' and events[1]['event']=='finished'
  if passed:assert events[1]['exit_code']==0
  for artifact,h in events[0]['artifact_sha256'].items():
   if artifact in p['tools']:assert p['tools'][artifact]==h;continue
   path=Path(artifact);assert path.parts[1:3]==('CantorAI','5090-study-20261023')
   relative=Path(*path.parts[3:]).as_posix();assert sha(z.read(relative))==h
  if events[1]['exit_code']==0:
   log=z.read(base+ledger+'/commands.log')
   if ledger=='cpu-ledger':assert b'Ran 10 tests' in log and b'\nOK' in log
   if ledger=='policy-ledger':assert b'BATCH_TOP1_POLICY_PASS parse=9 boundaries=112' in log
 for label,_,enabled in labels:
  n=base+label+'-observations.jsonl'
  if n in manifest['files']:
   if passed:replayed.append({'label':label,**replay(z.read(n),fixture,enabled)})
   else:failed_raw.append({'label':label,'bytes':len(z.read(n)),'sha256':sha(z.read(n))})
  else:assert not passed
  monitor_path=base+label+'-resources/resource-monitor.json'
  if monitor_path in manifest['files']:
   monitor=doc(monitor_path);samples=[json.loads(line) for line in z.read(base+label+'-resources/resource-vram.jsonl').splitlines()]
   peak=max(x['used_bytes'] for x in samples);peaks[label]=peak
   assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
   assert monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
   variant=next(v for v in w['variants'] if v['label']==label)
   assert monitor['child_exit_code']==variant['exit_code']
  else:assert not passed
 archived=sha(z.read(methods+'verify_integrated_batch_top1.py'))
report={'status':'integrated_batch_top1_correctness_replayed' if passed else 'integrated_batch_top1_failure_preserved',
 'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'variants':replayed,'failed_raw_preserved':failed_raw,
 'observed_unused_library_sha256':p['observed_unused_library_sha256'],'correctness_sampled_peak_adapter_bytes':peaks,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
 'replay_environment':{'python':platform.python_version(),'numpy':np.__version__},'native_runtime_built':False,
 'model_quality_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
