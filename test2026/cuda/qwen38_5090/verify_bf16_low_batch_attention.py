"""Independent FP64 replay for the isolated B1 attention candidate, not a model."""
from pathlib import Path
import hashlib,json,platform,sys,zipfile
import numpy as np
from bf16_low_batch_attention_oracle import ATOL,RTOL,case_plan,sample_triples,compare_witness,reference_case

out=Path(sys.argv[1]);name='bf16-low-batch-attention-native-attempt1';archive=out/(name+'-evidence.zip')
base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';sha=lambda b:hashlib.sha256(b).hexdigest()
numerical=[];failures=[];different_fp64=[]
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');collected=w['exit_code']==0
 prepraw=z.read(methods+'evidence/bf16-low-batch-attention-preparation.json');p=json.loads(prepraw)
 assert sha(prepraw)==w['preparation_sha256'] and w['source_sha256']==w['source_after_sha256']==p['files']
 assert w['inherited_model_sources']==w['inherited_model_sources_after']==p['inherited_model_sources']
 assert w['tools']==w['tools_after']==p['tools'] and w['library_unchanged'] and w['cleanup_finished'] and not w['runtime_loaded']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 for n,h in p['inherited_model_sources'].items():assert sha(z.read('Garnet/'+n))==h
 for helper in ('bf16_low_batch_attention_oracle.py','bf16_kv_overlay_oracle.py'):
  assert sha(Path(__file__).with_name(helper).read_bytes())==p['files'][methods+helper]
 parent=p['parent'];raw=z.read('results/'+parent+'-verified.json');proof=json.loads(raw)
 assert sha(raw)==p['parent_proof_sha256']==w['parent_proof_sha256']
 assert proof['status']=='bf16_padded_lifecycle_replayed' and len(proof['cases'])==14 and proof['raw_hidden_and_normalized_arrays']==102 and not proof['performance_admitted']
 assert proof['library_sha256']==w['observed_unused_library_sha256']==p['observed_unused_library_sha256']
 assert sha((out/(parent+'-evidence.zip')).read_bytes())==proof['archive_sha256']==p['parent_archive_sha256']==w['parent_archive_sha256']==manifest['parent_archive_sha256']
 owner=doc('tools/bf16-kv-native/low-batch-study-owner.json');assert Path(owner['study'])==Path('D:/CantorAI/5090-study-20261023') and owner['component']=='bf16-low-batch-attention'
 assert sha(z.read('tools/bf16-kv-native/bf16_kv_fixture_layout.h'))==p['fixture_header_sha256']
 if 'executable_sha256' in w:
  assert sha(z.read('tools/bf16-kv-native/bf16-low-batch-attention.exe'))==w['executable_sha256']==w['executable_after_sha256']
  assert Path(z.read('tools/bf16-kv-native/low-batch-host-compiler.txt').decode().splitlines()[0])==Path(p['host_compiler'])
 offline=doc(methods+'evidence/bf16-low-batch-attention-offline-preparation.json');plan=case_plan();assert plan==offline['case_plan'] and len(plan)==44
 for n,h in offline['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 for target,recipe in offline['files'].items():
  source=z.read('Garnet/'+recipe['source']);assert sha(source)==recipe['source_sha256'];text=source.decode().replace('\r\n','\n')
  for before,after,count in recipe['replacements']:assert text.count(before)==count;text=text.replace(before,after)
  assert text.encode()==z.read(methods+target) and sha(text.encode())==recipe['result_sha256']
 frozen=doc(methods+'evidence/bf16-low-batch-fp64-reference-values.json')
 assert frozen['status']=='low_batch_fp64_cpu_reference_values_frozen_gpu_unrun' and not frozen['gpu_executed'] and frozen['model_outputs']==0
 assert [r['case'] for r in frozen['cases']]==plan and frozen['bytes']==119040*8
 for n,h in frozen['oracle_sha256'].items():assert sha(z.read(methods+n))==h
 events={}
 for ledger in ('cpu-ledger','build-ledger','gpu-ledger'):
  n=base+ledger+'/commands.jsonl'
  if n not in manifest['files']:assert not collected;continue
  e=[json.loads(line) for line in z.read(n).splitlines()];assert len(e)==2 and e[0]['event']=='started' and e[-1]['event']=='finished';events[ledger]=e[-1]['exit_code']
  for artifact,h in e[0]['artifact_sha256'].items():
   if artifact in p['tools']:assert h==p['tools'][artifact];continue
   path=Path(artifact);assert path.parts[1:3]==('CantorAI','5090-study-20261023');relative=Path(*path.parts[3:]).as_posix();assert sha(z.read(relative))==h
 if collected:
  assert w['status']=='bf16_low_batch_attention_collected_needs_fp64_replay'
  assert events=={'cpu-ledger':0,'build-ledger':0,'gpu-ledger':0}
  assert b'Ran 10 tests' in z.read(base+'cpu-ledger/commands.log') and b'\nOK' in z.read(base+'cpu-ledger/commands.log')
  assert b'LOW_BATCH_CPU_PROFILE_PASS cases=33' in z.read(base+'build-ledger/commands.log')
  imports=z.read('tools/bf16-kv-native/low-batch-policy-imports.txt').lower()
  assert not any(n in imports for n in (b'cudart',b'nvcuda',b'cublas',b'nvinfer',b'garnet.dll'))
  records=[]
  for line in z.read(base+'gpu-ledger/commands.log').splitlines():
   if not line.startswith(b'{'):continue
   item=json.loads(line)
   if 'event' in item or 'status' in item:records.append(item)
  assert len(records)==90 and records[0]['event']=='gpu_identity' and records[0]['sm']==120 and records[0]['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
  assert records[-1]=={'status':'bf16_low_batch_overlay_collected_needs_fp64_replay','cases':44,'witness_bytes':119040*4,'performance_admitted':False,'model_qualified':False}
  binary=z.read(base+'witnesses.f32');assert len(binary)==119040*4;cursor=0
  for i,case in enumerate(plan):
   started,record=records[1+2*i:3+2*i]
   assert started=={'event':'case_started',**case} and record['event']=='case_collected'
   assert all(record[k]==v for k,v in case.items())
   triples=sample_triples(case);length=len(triples)*256*4
   assert record['witness_offset']==cursor and record['witness_bytes']==length and record['samples']==[list(t) for t in triples]
   assert record['readonly_exact'] and record['canaries_exact'] and record['prefix_cache_exact']==bool(case['end'])
   assert record['graph_replay_exact']==(case['id'] in (1,15,63)) and record['rejected_calls']==(52 if case['id'] in (0,14) else 0)
   raw=binary[cursor:cursor+length];cursor+=length
   try:numerical.append({'id':case['id'],**compare_witness(case,raw)})
   except ValueError as error:failures.append({'id':case['id'],'error':str(error)})
   values=reference_case(case)
   dense=np.stack([values[t] for t in triples]).reshape(-1).astype('<f8').tobytes() if triples else b''
   if sha(dense)!=frozen['cases'][i]['sha256']:different_fp64.append(case['id'])
  assert cursor==len(binary)
 peak=None
 if base+'resources/resource-monitor.json' in manifest['files']:
  monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()]
  peak=max(s['used_bytes'] for s in samples)
  assert monitor['child_exit_code']==w['exit_code'] and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
  assert monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
 else:assert not collected
 archived=sha(z.read(methods+'verify_bf16_low_batch_attention.py'))
passed=collected and not failures
report={'status':'bf16_low_batch_attention_fp64_replayed' if passed else 'bf16_low_batch_attention_failure_preserved',
 'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'observed_unused_library_sha256':p['observed_unused_library_sha256'],
 'gpu_exit_code':w['exit_code'],'cases':numerical,'numerical_failures':failures,'predeclared_reference_fp64_different_vector_hash_ids':different_fp64,
 'atol':ATOL,'rtol':RTOL,'correctness_sampled_peak_adapter_bytes':peak,'replay_environment':{'python':platform.python_version(),'numpy':np.__version__},
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
 'native_model_profile_admitted':False,'model_quality_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
raise SystemExit(0 if passed else 1)
