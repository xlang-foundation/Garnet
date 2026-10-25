"""Prepare standalone B1 collection only after the real model worker is replayed."""
from pathlib import Path
import argparse,hashlib,json,math,sys,unittest,zipfile

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('bf16_low_batch_commit_candidate.h','bf16_low_batch_commit_candidate.cu','probe_bf16_low_batch_commit.cu',
 'probe_bf16_low_batch_commit_policy.cpp','bf16_low_batch_layout.py','bf16_low_batch_commit_oracle.py','test_bf16_low_batch_commit.py',
 'test_kv_bf16_study_layout.py','probe_bf16_kv_commit_native.cu','compile_bf16_low_batch_commit.bat',
 'run_bf16_low_batch_commit.py','verify_bf16_low_batch_commit.py','prepare_bf16_low_batch_commit_runtime.py',
 'replay_bf16_low_batch_commit_observations.py','test_bf16_low_batch_commit_observations.py',
 'run_logged.py','monitor_native_probe.py','sample_vram.py',
 'evidence/bf16-low-batch-commit-source-derivation.json','evidence/bf16-low-batch-commit-probe-derivation.json',
 'evidence/bf16-low-batch-commit-byte-reference.json')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proofs',type=Path,required=True);parser.add_argument('--expected',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';model='integrated-batch-top1-model-lifecycle-attempt1';attention='bf16-low-batch-attention-native-attempt1';commit='bf16-kv-commit-native-attempt1'
 required=[args.proofs/(n+s) for n in (model,attention) for s in ('-verified.json','-remote-verified.json','-evidence.zip')]+[args.proofs/(commit+s) for s in ('-verified.json','-evidence.zip')]+[args.expected]
 missing=[str(p) for p in required if not p.is_file()]
 if missing:print(json.dumps({'status':'b1_commit_prerequisites_not_ready_no_preparation','missing':missing}));return 2
 local=json.loads((args.proofs/(model+'-verified.json')).read_bytes());raw=(args.proofs/(model+'-remote-verified.json')).read_bytes();proof=json.loads(raw)
 assert local==proof and proof['status'] in ('integrated_top1_full_model_lifecycle_replayed','integrated_top1_full_model_lifecycle_failure_preserved') and not proof['performance_admitted']
 assert sha(args.proofs/(model+'-evidence.zip'))==proof['archive_sha256']
 assert proof['replay_verifier_sha256']==proof['archived_replay_verifier_sha256']==sha(m/'verify_integrated_batch_top1_model.py')
 if proof['status']=='integrated_top1_full_model_lifecycle_replayed':assert len(proof['variants'])==2 and all(v['cases']==14 and v['raw_vectors_exact']==102 for v in proof['variants'])
 parents={model:{'status':proof['status'],'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':proof['archive_sha256']}}
 with zipfile.ZipFile(args.proofs/(model+'-evidence.zip')) as z:
  w=json.loads(z.read('results/'+model+'/workflow.json'))
  assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']
  assert w['runtime_dependencies']==w['runtime_dependencies_after'] and w['library_sha256']==proof['library_sha256']==w['runtime_dependencies']['garnet.dll']
  assert (w['exit_code']==0)==(proof['status']=='integrated_top1_full_model_lifecycle_replayed')
  inherited=w['source_sha256'];dependencies=w['runtime_dependencies']
 # Both actual numerical replays must pass; retain declared FP64 last-bit and
 # environment differences rather than demanding false bitwise identity.
 a=json.loads((args.proofs/(attention+'-verified.json')).read_bytes());raw=(args.proofs/(attention+'-remote-verified.json')).read_bytes();b=json.loads(raw)
 excluded={'cases','predeclared_reference_fp64_different_vector_hash_ids','replay_environment'}
 assert {k:v for k,v in a.items() if k not in excluded}=={k:v for k,v in b.items() if k not in excluded}
 for check in (a,b):
  assert check['status']=='bf16_low_batch_attention_fp64_replayed' and not check['numerical_failures'] and not check['performance_admitted']
  assert len(check['cases'])==44 and sum(c['scalars'] for c in check['cases'])==119040
  assert check['atol']==3e-5 and check['rtol']==2e-5 and all(math.isfinite(c['max_tolerance_fraction']) and 0<=c['max_tolerance_fraction']<=1 for c in check['cases'])
 assert [(c['id'],c['samples'],c['scalars']) for c in a['cases']]==[(c['id'],c['samples'],c['scalars']) for c in b['cases']]
 assert sha(args.proofs/(attention+'-evidence.zip'))==b['archive_sha256']
 parents[attention]={'status':b['status'],'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':b['archive_sha256']}
 raw=(args.proofs/(commit+'-verified.json')).read_bytes();c=json.loads(raw);assert c['status']=='bf16_kv_commit_native_operator_replayed' and c['full_state_byte_cases']==34 and c['rejected_calls']==62 and not c['performance_admitted']
 assert sha(args.proofs/(commit+'-evidence.zip'))==c['archive_sha256'];parents[commit]={'status':c['status'],'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':c['archive_sha256']}
 recipe=json.loads((e/'bf16-low-batch-commit-source-derivation.json').read_bytes())
 files={}
 for target,entry in recipe['files'].items():
  source=repo/entry['source'];assert sha(source)==entry['source_sha256'];files['Garnet/'+entry['source']]=entry['source_sha256']
  text=source.read_bytes().decode().replace('\r\n','\n')
  for before,after,count in entry['changes']:assert text.count(before)==count;text=text.replace(before,after)
  assert text.encode()==(m/target).read_bytes() and sha(m/target)==entry['result_sha256']
 with zipfile.ZipFile(args.proofs/(commit+'-evidence.zip')) as z:
  n='Garnet/src/cuda/qwen38_bf16_kv_commit.cu';assert hashlib.sha256(z.read(n)).hexdigest()==files[n]
 compiled=json.loads((e/'bf16-low-batch-commit-local-build.json').read_bytes());assert compiled['status']=='b1_commit_cpu_policy_oracle_and_private_compile_passed_gpu_unrun' and not compiled['gpu_executed']
 for n,h in compiled['input_sha256'].items():assert sha(Path(n))==h
 frozen=json.loads((e/'bf16-low-batch-commit-byte-reference.json').read_bytes());assert not frozen['gpu_executed'] and frozen['bytes']==104704 and sha(args.expected)==frozen['sha256']
 files.update({'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in METHODS})
 files['Garnet/src/cuda/qwen38_bf16_kv_commit.cuh']=sha(repo/'src/cuda/qwen38_bf16_kv_commit.cuh')
 files['tools/bf16-kv-native/bf16_kv_fixture_layout.h']=compiled['fixture_header_sha256']
 files['results/bf16-low-batch-commit-expected.bin']=frozen['sha256']
 toolchain=json.loads((e/'integrated-batch-top1-preparation.json').read_bytes())
 sys.path.insert(0,str(repo));test_ids=[]
 def collect(suite):
  for test in suite:
   if isinstance(test,unittest.TestSuite):yield from collect(test)
   else:yield test.id()
 for module in ('test_bf16_low_batch_commit','test_bf16_low_batch_commit_observations','test_kv_bf16_study_layout'):
  suite=unittest.defaultTestLoader.loadTestsFromName('test2026.cuda.qwen38_5090.'+module);test_ids.extend(collect(suite))
 assert len(test_ids)==len(set(test_ids)) and not any('FailedTest' in n for n in test_ids)
 p={'status':'b1_commit_prepared_unrun','files':files,'inherited_model_sources':inherited,'runtime_dependencies':dependencies,'parents':parents,'test_ids':sorted(test_ids),
  'tools':toolchain['tools'],'vcvars':toolchain['vcvars'],'nvcc':toolchain['nvcc'],'host_compiler':toolchain['host_compiler'],'fixture_header_sha256':compiled['fixture_header_sha256'],
  'scope':'Standalone B1/T4,T32 commit only after the original model worker is terminal, cleaned and independently replayed on both hosts. A replayed failed model parent stays FAILED; operator success never promotes it.20predeclared raw cases/104704frozen bytes, full host/GPU canonical comparisons,3graph profiles/3replays each,66negative calls/all7unused span rows and readonly/canary controls. Actual pinned compiler; no Garnet DLL load/build/replacement or model/profile/plugin/quality/4K/HTTP/speed/2x admission.'}
 target=e/'bf16-low-batch-commit-preparation.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 plan={'status':'b1_commit_upload_plan_prepared_unrun','uploads':[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS],
  'expected':{'local':str(args.expected),'target':'results/bf16-low-batch-commit-expected.bin','sha256':sha(args.expected)},
  'preparation':{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)},
  'scope':'Perform read-only remote preflight on every target, source, parent, compiler, unused runtime and fixed scratch ownership. Preserve unknown bytes, upload absent/matching methods only, acknowledge every upload, then one original controller. Never run beside the active model worker.'}
 data=json.dumps(plan,indent=2).encode()
 if args.plan.exists():assert args.plan.read_bytes()==data
 else:args.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'preparation_sha256':sha(target),'model_parent_status':proof['status'],'methods':len(METHODS)}));return 0

if __name__=='__main__':raise SystemExit(main())
