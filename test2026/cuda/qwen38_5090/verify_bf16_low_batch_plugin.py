"""Independent source/archive and exact same-kernel forwarding replay, no model claim."""
from pathlib import Path
import hashlib,json,re,sys,zipfile
from replay_bf16_low_batch_plugin_attention import records,compare
out=Path(sys.argv[1]);name='bf16-low-batch-plugin-native-attempt1';archive=out/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
checked=None;failures=[];contract=None;peak=None
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');raw=z.read(methods+'evidence/bf16-low-batch-plugin-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files']
 assert w['inherited_model_sources']==w['inherited_model_sources_after']==p['inherited_model_sources']
 assert w['tools']==w['tools_after']==p['tools'] and w['sdk_sha256']==w['sdk_after_sha256']==p['sdk_sha256']
 assert w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies'] and w['library_unchanged'] and w['cleanup_finished'] and w['garnet_runtime_loaded'] is False
 assert w['observed_unused_library_sha256']==p['runtime_dependencies']['garnet.dll'] and manifest['parents']==w['parents']==p['parents']
 for field in ('files','inherited_model_sources'):
  for n,h in p[field].items():assert sha(z.read(n))==h
 assert sha(Path(__file__).with_name('replay_bf16_low_batch_plugin_attention.py').read_bytes())==p['files'][methods+'replay_bf16_low_batch_plugin_attention.py']
 for parent,entry in p['parents'].items():
  raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw)
  assert sha(raw)==entry['proof_sha256'] and proof['status']==entry['status'] and sha((out/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256']==entry['archive_sha256']
 model='integrated-batch-top1-model-lifecycle-attempt1';modelproof=doc('results/'+model+'-verified.json')
 assert modelproof['status'] in ('integrated_top1_full_model_lifecycle_replayed','integrated_top1_full_model_lifecycle_failure_preserved') and modelproof['library_sha256']==w['observed_unused_library_sha256']
 for parent in (model,'bf16-low-batch-commit-native-attempt1'):
  with zipfile.ZipFile(out/(parent+'-evidence.zip')) as parent_zip:
   previous=json.loads(parent_zip.read('results/'+parent+'/workflow.json'));assert previous['cleanup_finished'] and previous['library_unchanged']
   assert previous['runtime_dependencies']==previous['runtime_dependencies_after']==p['runtime_dependencies']
   if parent==model:assert previous['source_sha256']==previous['source_after_sha256']==p['inherited_model_sources']
   else:assert previous['exit_code']==0 and previous['inherited_model_sources']==previous['inherited_model_sources_after']==p['inherited_model_sources']
 parent='bf16-low-batch-attention-native-attempt1';parentproof=doc('results/'+parent+'-verified.json');assert parentproof['status']=='bf16_low_batch_attention_fp64_replayed' and not parentproof['numerical_failures'] and len(parentproof['cases'])==44
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as parent_zip:
  parent_base='results/'+parent+'/';parent_records=records(parent_zip.read(parent_base+'gpu-ledger/commands.log'));parent_binary=parent_zip.read(parent_base+'witnesses.f32')
  for n in ('bf16_kv_low_batch_candidate.h','bf16_kv_low_batch_candidate.cu','probe_bf16_low_batch_attention.cu'):assert parent_zip.read(methods+n)==z.read(methods+n)
 # Exact source transformations; no broadening of the old B4/B8 wrapper.
 wrapper=doc(methods+'evidence/bf16-low-batch-overlay-plugin-source-derivation.json');entries=list(wrapper['files'].items())
 entries += [(target,doc(methods+'evidence/'+recipe)) for target,recipe in [('probe_bf16_low_batch_plugin_contract.cpp','bf16-low-batch-plugin-cpu-probe-derivation.json'),('probe_bf16_low_batch_plugin_attention.cu','bf16-low-batch-plugin-gpu-probe-derivation.json')]]
 for target,entry in entries:
  raw=z.read('Garnet/'+entry['source']);assert sha(raw)==entry['source_sha256'];text=raw.decode().replace('\r\n','\n')
  for before,after,count in entry['changes']:assert text.count(before)==count;text=text.replace(before,after)
  assert text.encode()==z.read(methods+target) and sha(text.encode())==entry['result_sha256']
 adapter=doc(methods+'evidence/bf16-low-batch-plugin-cpu-adapter-derivation.json')
 def extract(text,signature):
  start=text.index(signature);a=text.index('{',start);level=1;i=a+1
  while level:level+=(text[i]=='{')-(text[i]=='}');i+=1
  return text[start:i]
 expected='#include "bf16_kv_low_batch_candidate.h"\n#include "qwen38_bf16_kv_overlay.cuh"\nnamespace Garnet {\nnamespace { int unexpected=0;\n'
 for tag in ('Legacy','LowBatch'):
  row=adapter['source_functions'][tag];raw=z.read('Garnet/'+row['source']);assert sha(raw)==row['source_sha256'];body=extract(raw.decode().replace('\r\n','\n'),'bool valid(')
  assert body==row['valid_function'];renamed=body.replace('bool valid(','bool valid'+tag+'(',1);assert renamed==row['renamed_valid_function'];expected+=renamed+'\n'
 expected+='}\n'
 for tag,signature in [('Legacy','size_t Qwen38BF16KVOverlayWorkspace('),('LowBatch','size_t Qwen38BF16KVLowBatchCandidateWorkspace(')]:
  row=adapter['source_functions'][tag];body=extract(z.read('Garnet/'+row['source']).decode().replace('\r\n','\n'),signature);assert body==row['workspace_function'] and body.count('valid(c)')==1
  renamed=body.replace('valid(c)','valid'+tag+'(c)');assert renamed==row['renamed_workspace_function'];expected+=renamed+'\n'
 expected+='int BF16LowBatchPluginCPUUnexpectedEnqueues() noexcept{return unexpected;}\n'
 for symbol in ('RunQwen38BF16KVOverlay','RunQwen38BF16KVLowBatchCandidate'):expected+='int '+symbol+'(const Qwen35AttentionCoreConfig&,const void* const*,void*,void*,int64_t,int64_t,cudaStream_t) noexcept{++unexpected;return 1;}\n'
 expected+='}\n';assert expected.encode()==z.read(methods+'bf16_low_batch_plugin_cpu_operators.cpp') and sha(expected.encode())==adapter['result_sha256']
 owner=doc('tools/bf16-kv-native/low-batch-plugin-owner.json');assert Path(owner['study'])==Path('D:/CantorAI/5090-study-20261023') and owner['component']=='bf16-low-batch-plugin'
 assert sha(z.read('tools/bf16-kv-native/bf16_kv_fixture_layout.h'))==p['fixture_header_sha256']
 if 'executable_sha256' in w:
  assert w['executable_sha256']==w['executables_after_sha256'] and len(w['executable_sha256'])==2
  for n,h in w['executable_sha256'].items():assert sha(z.read('tools/bf16-kv-native/'+n))==h
  assert Path(z.read('tools/bf16-kv-native/low-batch-plugin-host-compiler.txt').decode().splitlines()[0])==Path(p['host_compiler'])
 collected=w['exit_code']==0;events={}
 for ledger in ('checker-ledger','build-ledger','contract-ledger','gpu-ledger'):
  path=base+ledger+'/commands.jsonl'
  if path not in manifest['files']:assert not collected;continue
  entries=[json.loads(line) for line in z.read(path).splitlines()];assert len(entries)==2 and entries[0]['event']=='started' and entries[-1]['event']=='finished' and entries[0]['id']==entries[-1]['id'];events[ledger]=entries[-1]['exit_code']
  if collected:assert entries[-1]['exit_code']==0
  for artifact,h in entries[0]['artifact_sha256'].items():
   if artifact in p['tools']:assert h==p['tools'][artifact];continue
   path=Path(artifact);assert path.parts[1:3]==('CantorAI','5090-study-20261023');relative=Path(*path.parts[3:]).as_posix()
   if relative in p['sdk_sha256']:assert h==p['sdk_sha256'][relative]
   else:assert sha(z.read(relative))==h
 if events.get('contract-ledger')==0:
  observations=[json.loads(line) for line in z.read(base+'contract-ledger/commands.log').splitlines() if line.startswith(b'{') and b'"status"' in line];assert len(observations)==1
  contract=observations[0];expected={'status':'bf16_low_batch_plugin_cpu_contract_passed','profiles':32,'roundtrips':32,'clones':32,'bad_wires':1632,'bad_descriptors':992,'bad_formats':672,'cross_dialect_rejections':96,'legacy_roundtrips':64,'unexpected_operator_calls':0,'runtime_version':101601,'header_version':101601,'loaded_tensorrt_dll':'D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll','enqueue_tested':False,'model_qualified':False,'performance_admitted':False}
  assert json.dumps(contract,sort_keys=True)==json.dumps(expected,sort_keys=True)
  imports=z.read('tools/bf16-kv-native/low-batch-plugin-cpu-imports.txt').lower();assert b'nvinfer_10.dll' in imports and not any(n in imports for n in (b'cudart',b'nvcuda',b'cublas',b'garnet.dll'))
 if collected:
  assert w['status']=='bf16_low_batch_plugin_collected_needs_exact_parent_replay' and events=={'checker-ledger':0,'build-ledger':0,'contract-ledger':0,'gpu-ledger':0}
  assert b'Ran 6 tests' in z.read(base+'checker-ledger/commands.log') and b'\nOK' in z.read(base+'checker-ledger/commands.log')
  try:checked=compare(parent_records,parent_binary,records(z.read(base+'gpu-ledger/commands.log')),z.read(base+'witnesses.f32'))
  except ValueError as error:failures.append({'error':str(error)})
 monitor_path=base+'resources/resource-monitor.json'
 if monitor_path in manifest['files']:
  monitor=doc(monitor_path);samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples)
  assert monitor['child_exit_code']==w['exit_code'] and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
  assert monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
 else:assert not collected
 archived=sha(z.read(methods+'verify_bf16_low_batch_plugin.py'))
passed=collected and not failures
report={'status':'bf16_low_batch_plugin_exact_parent_replayed' if passed else 'bf16_low_batch_plugin_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'observed_unused_library_sha256':w['observed_unused_library_sha256'],'closed_parent_model_status':modelproof['status'],'cpu_contract':contract,'forwarding':checked,'forwarding_failures':failures,'correctness_sampled_peak_adapter_bytes':peak,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'native_helper_lowering_model_admitted':False,'model_quality_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 1)
