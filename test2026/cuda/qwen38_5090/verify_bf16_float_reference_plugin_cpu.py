"""Replay reference plugin CPU assertions; valid TensorRT enqueue still untested."""
from pathlib import Path
import hashlib,json,sys,zipfile
out=Path(sys.argv[1]);name='bf16-float-reference-plugin-cpu-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');raw=z.read(methods+'evidence/bf16-float-reference-plugin-cpu-preparation.json');p=json.loads(raw)
 assert sha(raw)==w['preparation_sha256'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']==p['files'] and w['sdk_sha256']==w['sdk_after_sha256']==p['sdk_sha256']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 parent=doc('results/bf16-kv-float-reference-native-attempt1-verified.json')
 assert parent['status']=='bf16_float_reference_native_fp64_replayed' and len(parent['cases'])==42 and parent['fp64_witness_scalars']==442880 and parent['maximum_tolerance_fraction']<=1
 assert parent['archive_sha256']==w['parent_archive_sha256']==sha((out/'bf16-kv-float-reference-native-attempt1-evidence.zip').read_bytes())
 receipt=doc(methods+'evidence/bf16-float-reference-plugin-cpu-local-build.json')
 native_receipt=doc(methods+'evidence/bf16-kv-float-reference-probe-local-compile.json')
 for n,h in receipt['reused_object_sha256'].items():assert native_receipt['object_sha256'][n]==h
 for n,h in receipt['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 assert sha(z.read('tools/bf16-kv-native/probe_bf16_float_reference_plugin_cpu.exe'))==receipt['executable_sha256']
 assert sha(z.read(methods+'evidence/bf16-float-reference-plugin-cpu-build.log'))==receipt['build_log_sha256']
 assert sha(z.read(methods+'evidence/build_bf16_float_reference_plugin_cpu.bat.txt'))==receipt['recipe_sha256']
 events=[json.loads(l) for l in z.read(base+'cpu/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==w['exit_code']
 records=[]
 for line in z.read(base+'cpu/commands.log').splitlines():
  try:r=json.loads(line)
  except (ValueError,UnicodeDecodeError):continue
  if isinstance(r,dict) and r.get('status')=='bf16_float_reference_plugin_cpu_passed':records.append(r)
 passed=w['exit_code']==0
 if passed:
  assert w['status']=='bf16_float_reference_plugin_cpu_finished_needs_replay' and len(records)==1;r=records[0]
  for k,v in {'profiles':2,'clones':2,'roundtrips':2,'bad_wires':72,'bad_constructors':60,'bad_formats':98,'bad_descriptors':118,'invalid_enqueue_controls':118,'runtime_version':101601,'header_version':101601,'gpu_enqueue_tested':False,'model_qualified':False,'performance_admitted':False}.items():assert r[k]==v
  assert r['loaded_tensorrt_dll'].casefold()=='D:/CantorAI/5090-study-20261023/tools/TensorRT/bin/nvinfer_10.dll'.casefold()
 archived=sha(z.read(methods+'verify_bf16_float_reference_plugin_cpu.py'))
result={'status':'bf16_float_reference_plugin_cpu_replayed' if passed else 'bf16_float_reference_plugin_cpu_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
 'contract':records[0] if passed else None,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
 'gpu_enqueue_tested':False,'model_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
