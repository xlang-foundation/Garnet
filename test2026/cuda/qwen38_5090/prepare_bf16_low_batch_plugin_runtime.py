"""Prepare the isolated B1 plugin collector only after real terminal replay pairs."""
from pathlib import Path
import argparse,hashlib,json,math,zipfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
METHODS=('bf16_low_batch_overlay_plugin.h','bf16_low_batch_overlay_plugin.cpp','bf16_low_batch_plugin_cpu_operators.cpp',
 'probe_bf16_low_batch_plugin_contract.cpp','probe_bf16_low_batch_plugin_attention.cu','bf16_kv_low_batch_candidate.h','bf16_kv_low_batch_candidate.cu',
 'probe_bf16_low_batch_attention.cu','probe_bf16_kv_plugin_contract.cpp','replay_bf16_low_batch_plugin_attention.py','test_bf16_low_batch_plugin_attention_replay.py',
 'compile_bf16_low_batch_plugin.bat','run_bf16_low_batch_plugin.py','verify_bf16_low_batch_plugin.py','prepare_bf16_low_batch_plugin_runtime.py','run_logged.py','monitor_native_probe.py','sample_vram.py',
 'evidence/bf16-low-batch-overlay-plugin-source-derivation.json','evidence/bf16-low-batch-plugin-cpu-adapter-derivation.json',
 'evidence/bf16-low-batch-plugin-cpu-probe-derivation.json','evidence/bf16-low-batch-plugin-gpu-probe-derivation.json')
SOURCES=('src/backends/tensorrt/qwen38_bf16_kv_overlay_plugin.h','src/backends/tensorrt/qwen38_bf16_kv_overlay_plugin.cpp',
 'src/cuda/qwen38_bf16_kv_overlay.cu','src/cuda/qwen38_bf16_kv_overlay.cuh','src/cuda/qwen38_bf16_kv_commit.cuh','src/cuda/qwen35_attention_core.cuh')
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proofs',required=True,type=Path);parser.add_argument('--fixture',required=True,type=Path);parser.add_argument('--plan',required=True,type=Path);a=parser.parse_args()
 m=Path(__file__).resolve().parent;repo=m.parents[2];e=m/'evidence';model='integrated-batch-top1-model-lifecycle-attempt1';commit='bf16-low-batch-commit-native-attempt1';attention='bf16-low-batch-attention-native-attempt1'
 required=[a.proofs/(n+s) for n in (model,commit,attention) for s in ('-verified.json','-remote-verified.json','-evidence.zip')]+[a.fixture]
 missing=[str(p) for p in required if not p.is_file()]
 if missing:print(json.dumps({'status':'b1_plugin_prerequisites_not_ready_no_preparation','missing':missing}));return 2
 parents={};proofs={}
 for n in (model,commit,attention):
  local=json.loads((a.proofs/(n+'-verified.json')).read_bytes());raw=(a.proofs/(n+'-remote-verified.json')).read_bytes();remote=json.loads(raw)
  if n!=attention:assert local==remote
  else:
   excluded={'cases','predeclared_reference_fp64_different_vector_hash_ids','replay_environment'}
   assert {k:v for k,v in local.items() if k not in excluded}=={k:v for k,v in remote.items() if k not in excluded}
   for q in (local,remote):
    assert q['status']=='bf16_low_batch_attention_fp64_replayed' and not q['numerical_failures'] and len(q['cases'])==44
    assert sum(c['scalars'] for c in q['cases'])==119040 and all(math.isfinite(c['max_tolerance_fraction']) and 0<=c['max_tolerance_fraction']<=1 for c in q['cases'])
   assert [(c['id'],c['samples'],c['scalars']) for c in local['cases']]==[(c['id'],c['samples'],c['scalars']) for c in remote['cases']]
  assert not remote['performance_admitted'] and sha(a.proofs/(n+'-evidence.zip'))==remote['archive_sha256']
  assert remote['replay_verifier_sha256']==remote['archived_replay_verifier_sha256']==sha(m/('verify_integrated_batch_top1_model.py' if n==model else 'verify_bf16_low_batch_commit.py' if n==commit else 'verify_bf16_low_batch_attention.py'))
  parents[n]={'status':remote['status'],'proof_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':remote['archive_sha256']};proofs[n]=remote
 assert proofs[model]['status'] in ('integrated_top1_full_model_lifecycle_replayed','integrated_top1_full_model_lifecycle_failure_preserved')
 assert proofs[commit]['status']=='bf16_low_batch_commit_exact_bytes_replayed' and not proofs[commit]['byte_failures'] and len(proofs[commit]['cases'])==20
 with zipfile.ZipFile(a.proofs/(model+'-evidence.zip')) as z:
  closed=json.loads(z.read('results/'+model+'/workflow.json'));assert closed['cleanup_finished'] and closed['library_unchanged']
  assert closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
  assert (closed['exit_code']==0)==(proofs[model]['status']=='integrated_top1_full_model_lifecycle_replayed')
  inherited=closed['source_sha256'];dependencies=closed['runtime_dependencies']
 with zipfile.ZipFile(a.proofs/(commit+'-evidence.zip')) as z:
  closed_commit=json.loads(z.read('results/'+commit+'/workflow.json'));assert closed_commit['exit_code']==0 and closed_commit['cleanup_finished'] and closed_commit['library_unchanged']
  assert closed_commit['inherited_model_sources']==closed_commit['inherited_model_sources_after']==inherited
  assert closed_commit['runtime_dependencies']==closed_commit['runtime_dependencies_after']==dependencies
  assert closed_commit['observed_unused_library_sha256']==proofs[model]['library_sha256']==dependencies['garnet.dll']
 with zipfile.ZipFile(a.proofs/(attention+'-evidence.zip')) as z:
  for n in ('bf16_kv_low_batch_candidate.h','bf16_kv_low_batch_candidate.cu','probe_bf16_low_batch_attention.cu'):
   assert z.read('Garnet/test2026/cuda/qwen38_5090/'+n)==(m/n).read_bytes()
 # Literal wrapper and probe derivations remain distinct from the existing type.
 wrapper=json.loads((e/'bf16-low-batch-overlay-plugin-source-derivation.json').read_bytes())
 entries=list(wrapper['files'].items())+[(name,json.loads((e/recipe).read_bytes())) for name,recipe in [('probe_bf16_low_batch_plugin_contract.cpp','bf16-low-batch-plugin-cpu-probe-derivation.json'),('probe_bf16_low_batch_plugin_attention.cu','bf16-low-batch-plugin-gpu-probe-derivation.json')]]
 for target,record in entries:
  raw=(repo/record['source']).read_bytes();assert hashlib.sha256(raw).hexdigest()==record['source_sha256'];text=raw.decode().replace('\r\n','\n')
  for before,after,count in record['changes']:assert text.count(before)==count;text=text.replace(before,after)
  assert text.encode()==(m/target).read_bytes() and sha(m/target)==record['result_sha256']
 local_build=json.loads((e/'bf16-low-batch-overlay-plugin-local-build.json').read_bytes());gpu_build=json.loads((e/'bf16-low-batch-plugin-forwarding-local-build.json').read_bytes())
 assert not local_build['runtime_executed'] and not gpu_build['gpu_executed']
 for built in (local_build,gpu_build):
  for n,h in built['source_sha256'].items():assert sha(Path(n))==h
 fixture_hash=sha(a.fixture);assert fixture_hash==local_build['source_sha256'][str(a.fixture.resolve())]
 previous=json.loads((e/'bf16-plugin-cpu-contract-preparation.json').read_bytes());sdk=previous['sdk_sha256']
 for suffix in ('include/NvInfer.h','include/NvInferRuntime.h','include/NvInferRuntimePlugin.h','include/NvInferVersion.h','lib/nvinfer_10.lib'):
  assert sdk['tools/TensorRT/'+suffix]==local_build['source_sha256'][str(Path('D:/CantorAI/ThirdPartySDK/TensorRT')/suffix)]
 assert sdk['tools/TensorRT/bin/nvinfer_10.dll']==dependencies['nvinfer_10.dll']
 toolchain=json.loads((e/'integrated-batch-top1-preparation.json').read_bytes());files={'Garnet/test2026/cuda/qwen38_5090/'+n:sha(m/n) for n in METHODS}
 files.update({'Garnet/'+n:sha(repo/n) for n in SOURCES});files['tools/bf16-kv-native/bf16_kv_fixture_layout.h']=fixture_hash
 p={'status':'b1_plugin_prepared_unrun','files':files,'inherited_model_sources':inherited,'sdk_sha256':sdk,'runtime_dependencies':dependencies,'parents':parents,'tools':toolchain['tools'],
  'vcvars':toolchain['vcvars'],'nvcc':toolchain['nvcc'],'host_compiler':toolchain['host_compiler'],'fixture_header_sha256':fixture_hash,
  'scope':'Isolated B1 plugin CPU32profile/registry/wire/cross-dialect contract then44case/119040scalar forwarding on the same qualified kernel,87host enqueues,3graphs,40cache equivalence cases.104negative calls remain direct-kernel controls. Requires actual closed/replayed model and B1commit, idleGPU and no study executable. Failed model stays failed. No Garnet DLL rebuild/install/load, helper/lowering/model/quality/4K/HTTP/timing/2x admission.'}
 target=e/'bf16-low-batch-plugin-preparation.json';data=json.dumps(p,indent=2).encode()
 if target.exists():assert target.read_bytes()==data
 else:target.write_bytes(data)
 plan={'status':'b1_plugin_upload_plan_prepared_unrun','uploads':[{'local':str(m/n),'target':'Garnet/test2026/cuda/qwen38_5090/'+n,'sha256':sha(m/n)} for n in METHODS],
  'preparation':{'local':str(target),'target':'Garnet/test2026/cuda/qwen38_5090/evidence/'+target.name,'sha256':sha(target)},
  'scope':'Only after original supervisor is terminal and real replays finish. Read-only preflight every target/tool/source/runtime/parent and fixed scratch outputs. Preserve unknown or existing bytes, acknowledge each absent/matching method upload; reuse tools/bf16-kv-native. No SDK/DLL copying or second GPU worker.'}
 data=json.dumps(plan,indent=2).encode()
 if a.plan.exists():assert a.plan.read_bytes()==data
 else:a.plan.write_bytes(data)
 print(json.dumps({'status':p['status'],'preparation_sha256':sha(target),'methods':len(METHODS),'parent_model_status':proofs[model]['status']}));return 0
if __name__=='__main__':raise SystemExit(main())
