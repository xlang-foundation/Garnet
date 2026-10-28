"""Materialize one native build only from actual closed API proofs and SDK observation."""
from pathlib import Path
import argparse,hashlib,json,sys,unittest,zipfile

sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
METHODS=('build_integrated_batch_top1_native.bat','run_integrated_batch_top1_native_build.py',
 'verify_integrated_batch_top1_native_build.py','prepare_integrated_batch_top1_native_build.py',
 'bf16_padded_state_gate.py','test_bf16_padded_state_gate.py',
 'evidence/nvfp4-config.json','evidence/bf16-float-reference-cpu-completion-preparation.json',
 'evidence/integrated-top1-native-build-environment.json','evidence/integrated-top1-native-build-environment-observer.py.txt',
 'evidence/integrated-batch-top1-correctness-attempt1-verified.json',
 'evidence/integrated-batch-top1-correctness-attempt1-remote-verified.json')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proofs',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path);args=parser.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];sys.path.insert(0,str(repo));e=m/'evidence';name='integrated-batch-top1-correctness-attempt1'
 a=json.loads((args.proofs/(name+'-verified.json')).read_bytes());raw=(args.proofs/(name+'-remote-verified.json')).read_bytes();b=json.loads(raw)
 assert {k:v for k,v in a.items() if k!='replay_environment'}=={k:v for k,v in b.items() if k!='replay_environment'}
 assert b['status']=='integrated_batch_top1_correctness_replayed' and not b['native_runtime_built'] and not b['performance_admitted']
 assert [v['label'] for v in b['variants']]==['disabled','enabled','unset','malformed']
 assert all(v['cases']==352 and v['raw_observations']==1066 and v['graphs']==2 and v['invalid_calls']==12 for v in b['variants'])
 assert b['replay_verifier_sha256']==b['archived_replay_verifier_sha256']==sha(m/'verify_integrated_batch_top1.py')
 archive=args.proofs/(name+'-evidence.zip');assert sha(archive)==b['archive_sha256']
 with zipfile.ZipFile(archive) as z:
  w=json.loads(z.read('results/'+name+'/workflow.json'));cp=json.loads(z.read('Garnet/test2026/cuda/qwen38_5090/evidence/integrated-batch-top1-preparation.json'))
  assert w['exit_code']==0 and w['cleanup_finished'] and w['library_unchanged'] and not w['runtime_loaded']
  assert w['source_sha256']==w['source_after_sha256']==cp['files']
 observed=json.loads((e/'integrated-top1-native-build-environment.json').read_bytes())
 assert observed['status']=='integrated_top1_native_build_environment_observed' and not observed['live_study_executables'] and observed['gpu_used_mib']<1024
 assert sha(e/'integrated-top1-native-build-environment-observer.py.txt')==observed['observer_sha256']
 assert observed['runtime_dependencies']==cp['runtime_dependencies'] and observed['runtime_dependencies']['garnet.dll']==b['observed_unused_library_sha256']
 previous=json.loads((e/'bf16-float-reference-cpu-completion-preparation.json').read_bytes());ids=set(previous['test_ids']);assert len(ids)==50
 modules=['test2026.cuda.qwen38_5090.'+n for n in ('test_parallel_top1_oracle','test_integrated_top1_oracle','test_bf16_padded_state_gate')]
 def collect(suite):
  for item in suite:
   if isinstance(item,unittest.TestSuite):yield from collect(item)
   else:yield item.id()
 for module in modules:ids.update(collect(unittest.defaultTestLoader.loadTestsFromName(module)))
 assert len(ids)==63
 files=dict(cp['files']);files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in METHODS})
 files['Garnet/test2026/cuda/qwen38_5090/CMakeLists.txt']=sha(m/'CMakeLists.txt')
 for n,h in files.items():
  if n.startswith('Garnet/'):assert sha(repo/n.removeprefix('Garnet/'))==h,('Local source/input drift',n)
 p={'status':'native_sampler_build_prepared_unrun','files':files,'component_sources':cp['files'],
  'component_proof_sha256':hashlib.sha256(raw).hexdigest(),'component_archive_sha256':b['archive_sha256'],
  'environment_files':observed['files'],'runtime_dependencies':observed['runtime_dependencies'],
  'cmake_cache_before_sha256':observed['cmake_cache_sha256'],'expected_cmake_cache':observed['cache'],
  'vcvars':cp['vcvars'],'test_ids':sorted(ids),
  'scope':'Actual preserved native Windows SDK Release/sm120/MT build after complete four-mode production API replay;63explicit CPU model/profile/sampler tests and9settings/112CPU policy boundaries. Preserve original97cDLL and allfive runtime dependencies, keep vendor DLL staging and whole-program CUDA disabled. No native GPU model, quality, HTTP memory/speed or2x qualification.'}
 target=e/'integrated-batch-top1-native-build-preparation.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 plan={'status':'native_build_method_upload_plan_prepared_unrun',
  'uploads':[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS],
  'preparation':{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)},
  'scope':'Read-only preflight every target: preserve existing matching parent inputs and reject unknown bytes; upload only absent methods/receipts/preparation, acknowledge all before original build controller. No source/runtime overwrite during a GPU worker.'}
 data=json.dumps(plan,indent=2).encode()
 if args.plan.exists():assert args.plan.read_bytes()==data
 else:args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'cpu_tests':len(ids),'source_and_method_files':len(files),'preparation_sha256':sha(target)}));return 0

if __name__=='__main__':raise SystemExit(main())
