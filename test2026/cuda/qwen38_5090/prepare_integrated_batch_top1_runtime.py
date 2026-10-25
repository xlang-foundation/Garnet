"""Local preparation from actual closed/replayed evidence; never invent readiness."""
from pathlib import Path
import argparse,hashlib,json,sys,zipfile

METHODS=('install_integrated_batch_top1_sources.py','verify_integrated_batch_top1_source_install.py',
 'observe_integrated_batch_top1_install.py','prepare_integrated_batch_top1_runtime.py',
 'probe_integrated_batch_top1.cu','probe_legacy_batch_top1.cu','probe_parallel_batch_top1.cu',
 'integrated_top1_oracle.py','test_integrated_top1_oracle.py','parallel_top1_oracle.py','test_parallel_top1_oracle.py',
 'replay_integrated_top1_observations.py','run_integrated_batch_top1.py','verify_integrated_batch_top1.py','compile_integrated_batch_top1.bat',
 'run_logged.py','monitor_native_probe.py','sample_vram.py','verify_bf16_padded_lifecycle_scope_fixed.py',
 'evidence/integrated-batch-top1-fixture-oracles.json','evidence/integrated-batch-top1-probe-derivation.json',
 'evidence/parallel-top1-fixture-oracles.json','evidence/parallel-batch-top1-legacy-sampler.cu.txt',
 'evidence/integrated-batch-top1-source-install-offline-preparation.json')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
raw_sha=lambda b:hashlib.sha256(b).hexdigest()

def write_once(path,raw):
 if path.exists():assert path.read_bytes()==raw,('Preserve different existing preparation',str(path))
 else:path.write_bytes(raw)

def inventory(methods):
 prefix='test2026/cuda/qwen38_5090/';known={}
 old=json.loads((methods/'evidence/bf16-padded-lifecycle-scope-fixed-preparation.json').read_bytes())
 known.update({prefix+n:h for n,h in old['methods'].items()})
 operator=json.loads((methods/'evidence/parallel-batch-top1-preparation.json').read_bytes())
 for n,h in operator['files'].items():
  if n.startswith('Garnet/'+prefix):
   name=n.removeprefix('Garnet/')
   if name in known:assert known[name]==h,('Conflicting known method',name)
   known[name]=h
 desired={prefix+n:sha(methods/n) for n in METHODS}
 return {'status':'prospective_install_method_inventory','desired_methods_sha256':desired,
         'known_existing_methods_sha256':{n:known[n] for n in desired if n in known},
         'scope':'CPU file inventory only; not parent completion, source installation or GPU/model qualification.'}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--inventory',type=Path,required=True)
 parser.add_argument('--inventory-only',action='store_true');parser.add_argument('--proofs',type=Path)
 parser.add_argument('--observation',type=Path);parser.add_argument('--packet',type=Path);parser.add_argument('--plan',type=Path)
 args=parser.parse_args();methods=Path(__file__).resolve().parent;repo=methods.parents[2]
 spec=inventory(methods);specraw=json.dumps(spec,indent=2).encode();write_once(args.inventory,specraw)
 if args.inventory_only:
  print(json.dumps({'status':'method_inventory_only_no_runtime_preparation','methods':len(METHODS),'sha256':raw_sha(specraw)}));return 0
 assert all((args.proofs,args.observation,args.packet,args.plan)),'Runtime preparation requires proofs, observation, packet and plan'
 parent='bf16-padded-lifecycle-scope-fixed-attempt2';operator='parallel-batch-top1-correctness-attempt1'
 required=[args.proofs/(n+suffix) for n in (parent,operator) for suffix in ('-verified.json','-remote-verified.json','-evidence.zip')]+[args.observation,args.packet]
 missing=[str(f) for f in required if not f.is_file()]
 if missing:
  print(json.dumps({'status':'prerequisites_not_ready_no_runtime_preparation','missing':missing}));return 2
 proofraw=(args.proofs/(parent+'-remote-verified.json')).read_bytes();proof=json.loads(proofraw)
 assert proof==json.loads((args.proofs/(parent+'-verified.json')).read_bytes())
 assert proof['status']=='bf16_padded_lifecycle_replayed' and len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102 and not proof['performance_admitted']
 assert proof['replay_verifier_sha256']==proof['archived_replay_verifier_sha256']==sha(methods/'verify_bf16_padded_lifecycle_scope_fixed.py')
 assert sha(args.proofs/(parent+'-evidence.zip'))==proof['archive_sha256']
 opraw=(args.proofs/(operator+'-remote-verified.json')).read_bytes();op=json.loads(opraw)
 localop=json.loads((args.proofs/(operator+'-verified.json')).read_bytes());left=dict(op);right=dict(localop)
 left.pop('replay_environment');right.pop('replay_environment');assert left==right
 assert op['status']=='parallel_batch_top1_correctness_replayed' and op['cases_checked']==330 and op['raw_observations_checked']==1000 and not op['performance_admitted']
 assert sha(args.proofs/(operator+'-evidence.zip'))==op['archive_sha256']
 observationraw=args.observation.read_bytes();observed=json.loads(observationraw)
 assert observed['status']=='integrated_top1_post_lifecycle_install_observed'
 assert observed['observer_sha256']==sha(methods/'observe_integrated_batch_top1_install.py') and observed['method_inventory_sha256']==raw_sha(specraw)
 assert observed['parent_proof_sha256']==raw_sha(proofraw) and observed['parent_archive_sha256']==proof['archive_sha256']
 assert set(observed['methods_before_sha256'])==set(spec['desired_methods_sha256'])
 for n,h in observed['methods_before_sha256'].items():
  assert h is None or h in (spec['desired_methods_sha256'][n],spec['known_existing_methods_sha256'].get(n))
 assert not observed['live_study_executables'] and observed['gpu_used_mib']<1024
 offline_raw=(methods/'evidence/integrated-batch-top1-source-install-offline-preparation.json').read_bytes();offline=json.loads(offline_raw)
 assert observed['offline_manifest_sha256']==raw_sha(offline_raw) and observed['base_source_sha256']==offline['base_source_sha256']
 assert sha(args.packet)==offline['source_zip_sha256'] and args.packet.stat().st_size==offline['source_zip_bytes']
 with zipfile.ZipFile(args.packet) as z:
  assert z.testzip() is None and set(z.namelist())==set(offline['candidate_source_sha256']) and len(z.namelist())==5
  for n,h in offline['candidate_source_sha256'].items():assert raw_sha(z.read(n))==sha(repo/n)==h
 with zipfile.ZipFile(args.proofs/(parent+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+parent+'/workflow.json'))
  assert w['exit_code']==0 and w['cleanup_finished'] and w['library_unchanged']
  assert w['source_sha256']==w['source_after_sha256']==observed['qualified_parent_source_sha256']
 with zipfile.ZipFile(args.proofs/(operator+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+operator+'/workflow.json'))
  assert w['executable_sha256']==w['executable_after_sha256']==observed['preserved_component_executable_sha256']
 for n,h in offline['base_source_sha256'].items():
  if n not in offline['candidate_source_sha256']:assert sha(repo/n)==h,('Local inherited source drift',n)
 assert observed['runtime_dependencies']['garnet.dll']==proof['library_sha256']==offline['required_observed_unused_library_sha256']
 install_names=('install_integrated_batch_top1_sources.py','verify_integrated_batch_top1_source_install.py',
   'observe_integrated_batch_top1_install.py','prepare_integrated_batch_top1_runtime.py','evidence/integrated-batch-top1-source-install-offline-preparation.json')
 install={'status':'source_install_prepared_unrun','source_zip_sha256':offline['source_zip_sha256'],
  'base_source_sha256':offline['base_source_sha256'],'candidate_source_sha256':offline['candidate_source_sha256'],'allowed_new_source':offline['allowed_new_source'],
  'qualified_parent_source_sha256':observed['qualified_parent_source_sha256'],'parent_proof_sha256':raw_sha(proofraw),'parent_archive_sha256':proof['archive_sha256'],
  'runtime_dependencies':observed['runtime_dependencies'],'methods':{n:sha(methods/n) for n in install_names},
  'scope':'Controlled five-file source-only installation after closed full14case/102vector parent and both independent replays. Preserve three original files, verify all base/runtime hashes, reject any live study executable. No native runtime build/load, SDK replacement, model or speed admission.'}
 installraw=json.dumps(install,indent=2).encode();installpath=methods/'evidence/integrated-batch-top1-source-install-preparation.json'
 sources=dict(offline['base_source_sha256']);sources.update(offline['candidate_source_sha256'])
 files={'Garnet/'+n:h for n,h in sources.items()};files.update({'Garnet/'+n:h for n,h in spec['desired_methods_sha256'].items()})
 files['Garnet/test2026/cuda/qwen38_5090/evidence/integrated-batch-top1-source-install-preparation.json']=raw_sha(installraw)
 files.update({'tools/integrated-top1-install-offline.json':raw_sha(offline_raw),'tools/integrated-top1-install-methods.json':raw_sha(specraw),
  'tools/observe-integrated-top1-install.py':observed['observer_sha256'],'results/integrated-top1-install-observation.json':raw_sha(observationraw)})
 p={'status':'integrated_top1_production_api_prepared_unrun','files':files,'tools':observed['tools'],'runtime_dependencies':observed['runtime_dependencies'],
  'qualified_parent_source_sha256':observed['qualified_parent_source_sha256'],'parent_proof_sha256':raw_sha(proofraw),'parent_archive_sha256':proof['archive_sha256'],
  'operator_proof_sha256':raw_sha(opraw),'operator_archive_sha256':op['archive_sha256'],'observed_unused_library_sha256':proof['library_sha256'],
  'preserved_component_executable_sha256':observed['preserved_component_executable_sha256'],'source_install_preparation_sha256':raw_sha(installraw),
  'installed_source_zip_sha256':offline['source_zip_sha256'],'installed_candidate_source_sha256':offline['candidate_source_sha256'],
  'vcvars':observed['vcvars'],'nvcc':observed['nvcc'],'host_compiler':observed['host_compiler'],
  'scope':'Four independent production API processes:0/1/unset/01;352cases/1066raw observations each against frozen legacy GPU and independent CPU oracle, both exact modes/fallback/capture/canary/readonly/negative controls. Source-only installation must be independently replayed first. No Garnet runtime DLL loaded/rebuilt, model quality, serving memory, performance or2x admission.'}
 runtimepath=methods/'evidence/integrated-batch-top1-preparation.json';runtimeraw=json.dumps(p,indent=2).encode()
 uploads=[]
 for n,h in spec['desired_methods_sha256'].items():
  if observed['methods_before_sha256'][n]!=h:uploads.append({'local':str(repo/n),'target':'Garnet/'+n,'sha256':h,'expected_before_sha256':observed['methods_before_sha256'][n]})
 uploads.extend([{'local':str(installpath),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+installpath.name,'sha256':raw_sha(installraw),'expected_before_sha256':None},
  {'local':str(runtimepath),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+runtimepath.name,'sha256':raw_sha(runtimeraw),'expected_before_sha256':None},
  {'local':str(args.packet),'target':'results/integrated-batch-top1-source.zip','sha256':sha(args.packet),'expected_before_sha256':None}])
 plan={'status':'runtime_upload_plan_prepared_unrun','uploads':uploads,'source_install_preparation_sha256':raw_sha(installraw),'component_preparation_sha256':raw_sha(runtimeraw),
  'scope':'Upload methods/receipts and source packet only, acknowledge each, then run guarded installer and both source-install replays before the production API GPU controller. Never directly upload candidate source over deployed files.'}
 # All prerequisites are checked before materializing either runtime preparation.
 write_once(installpath,installraw);write_once(runtimepath,runtimeraw);write_once(args.plan,json.dumps(plan,indent=2).encode())
 print(json.dumps({'status':plan['status'],'uploads':len(uploads),'source_install_preparation_sha256':raw_sha(installraw),'component_preparation_sha256':raw_sha(runtimeraw)}));return 0

if __name__=='__main__':raise SystemExit(main())
