"""Replay raw GPU/legacy/serial-CPU outputs against predeclared independent data."""
from pathlib import Path
import hashlib,json,platform,sys,zipfile
import numpy as np
from parallel_top1_oracle import SHAPES,expected_rows

out=Path(sys.argv[1]);name='parallel-batch-top1-correctness-attempt1'
archive=out/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
sha=lambda b:hashlib.sha256(b).hexdigest();mismatches=[];counts={};cases_checked=0;observations_checked=0;truncated_tail=None
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0
 prepraw=z.read(methods+'evidence/parallel-batch-top1-preparation.json');p=json.loads(prepraw)
 assert sha(prepraw)==w['preparation_sha256'] and p['files']==w['source_sha256']==w['source_after_sha256']
 assert p['inherited_model_sources']==w['inherited_model_sources']==w['inherited_model_sources_after']
 assert p['tools']==w['tools']==w['tools_after'] and w['cleanup_finished'] and w['library_unchanged'] and not w['runtime_loaded']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 for n,h in p['inherited_model_sources'].items():assert sha(z.read('Garnet/'+n))==h
 assert sha(Path(__file__).with_name('parallel_top1_oracle.py').read_bytes())==p['files'][methods+'parallel_top1_oracle.py']
 parent=p['parent'];parentraw=z.read('results/'+parent+'-verified.json');proof=json.loads(parentraw)
 assert sha(parentraw)==p['parent_proof_sha256']==w['parent_proof_sha256']
 assert sha((out/(parent+'-evidence.zip')).read_bytes())==p['parent_archive_sha256']==w['parent_archive_sha256']==proof['archive_sha256']==manifest['parent_archive_sha256']
 assert proof['status']==w['parent_status'] and proof['status'] in ('bf16_padded_lifecycle_replayed','bf16_padded_lifecycle_failure_preserved')
 assert w['observed_unused_library_sha256']==p['observed_unused_library_sha256']==proof['library_sha256']
 owner=doc('tools/parallel-batch-top1/study-owner.json');assert Path(owner['study'])==Path('D:/CantorAI/5090-study-20261023') and owner['component']=='parallel-batch-top1'
 if 'executable_sha256' in w:
  assert sha(z.read('tools/parallel-batch-top1/probe.exe'))==w['executable_sha256']==w['executable_after_sha256']
  assert Path(z.read('tools/parallel-batch-top1/host-compiler.txt').decode().splitlines()[0])==Path(p['host_compiler'])
 fixture=doc(methods+'evidence/parallel-top1-fixture-oracles.json')
 assert fixture['status']=='cpu_fixture_oracles_predeclared_unrun' and not fixture['gpu_run'] and fixture['gpu_observations_planned']==1000
 for n,h in fixture['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 expected_keys=[(dtype,rows,vocab,pattern) for dtype in ('fp32','bf16') for rows,vocab in SHAPES for pattern in range(11)]
 frozen={}
 for record in fixture['cases']:
  key=tuple(record[n] for n in ('dtype','rows','vocab','pattern'));assert key in expected_keys and key not in frozen
  ids,bits=expected_rows(*key);assert (record['ids'],record['score_bits'])==(ids,bits);frozen[key]=(ids,bits)
 assert list(frozen)==expected_keys and len(frozen)==330
 events={}
 for ledger in ('cpu-ledger','build-ledger','gpu-ledger'):
  n=base+ledger+'/commands.jsonl'
  if n not in manifest['files']:
   assert not passed;continue
  e=[json.loads(line) for line in z.read(n).splitlines()]
  assert len(e)==2 and e[0]['event']=='started' and e[-1]['event']=='finished'
  events[ledger]=e[-1]['exit_code']
  for artifact,h in e[0]['artifact_sha256'].items():
   # Inputs/binary were hashed before each command, not inferred afterward.
   if artifact in p['tools']:
    assert h==p['tools'][artifact];continue
   path=Path(artifact);parts=path.parts;assert parts[1]=='CantorAI' and parts[2]=='5090-study-20261023'
   relative=Path(*parts[3:]).as_posix()
   assert relative in manifest['files'] and sha(z.read(relative))==h
  if ledger=='cpu-ledger' and e[-1]['exit_code']==0:
   text=z.read(base+ledger+'/commands.log');assert b'Ran 7 tests' in text and b'\nOK' in text
 if passed:assert events=={'cpu-ledger':0,'build-ledger':0,'gpu-ledger':0}
 if base+'observations.jsonl' in manifest['files']:
  lines=z.read(base+'observations.jsonl').splitlines();records=[]
  for index,line in enumerate(lines):
   try:records.append(json.loads(line))
   except json.JSONDecodeError:
    assert not passed and index==len(lines)-1
    truncated_tail={'line_index':index,'bytes':len(line),'sha256':sha(line)}
  starts=[];completed=[];normal=[];extras=[];graphs=[];rejections=[];summaries=[]
  for record in records:
   kind=record['kind'];counts[kind]=counts.get(kind,0)+1
   if kind in ('case_started','observation','case'):
    key=tuple(record[n] for n in ('dtype','rows','vocab','pattern'));assert key in frozen
   if kind=='case_started':starts.append(key)
   elif kind=='observation':
    dtype,rows,vocab,pattern=key;ids,bits=frozen[key]
    assert all(type(n) is int for field in ('candidate_ids','legacy_ids','candidate_score_bits','legacy_score_bits','cpu_ids','cpu_score_bits') for n in record[field])
    assert len(record['candidate_ids'])==len(record['legacy_ids'])==len(record['candidate_score_bits'])==len(record['legacy_score_bits'])==rows+4
    common=(record['candidate_ids']==record['legacy_ids'] and record['candidate_score_bits']==record['legacy_score_bits'])
    cpu=(record['cpu_ids']==ids and record['cpu_score_bits']==bits)
    for mode in ('candidate','legacy'):
     gpu_ids=record[mode+'_ids'];gpu_bits=record[mode+'_score_bits']
     common=common and gpu_ids[2:-2]==ids and gpu_bits[2:-2]==bits
     common=common and all(gpu_ids[i]==0x5a5a5a5a5a5a5a5a and gpu_bits[i]==0x5a5a5a5a for i in (0,1,rows+2,rows+3))
    exact=common and cpu
    if not exact:mismatches.append({'key':list(key),'phase':record['phase'],'repeat':record['repeat'],'raw_observation_index':observations_checked})
    if passed:assert exact
    item=(key,record['phase'],record['repeat'])
    (normal if record['phase']=='normal' else extras).append(item);observations_checked+=1
   elif kind=='case':
    ids,bits=frozen[key];assert record['ids']==ids and record['score_bits']==bits
    assert record['repeats']==3 and record['cpu_legacy_candidate_exact'] and record['readonly'] and record['canaries']
    completed.append(key);cases_checked+=1
   elif kind=='graph':
    assert record['dtype'] in ('fp32','bf16') and record['rows']==4 and record['vocab']==248320 and record['replays']==3 and record['exact'] and record['nullable_score_exact'];graphs.append(record['dtype'])
   elif kind=='rejection':
    if passed:assert record['status']==record['expected']==1
    rejections.append((record['dtype'],record['label'],record['status'],record['expected']))
   elif kind=='summary':summaries.append(record)
   else:raise AssertionError('Unknown raw observation record kind')
  if passed:
   assert starts==completed==expected_keys and len(records)==1675
   assert normal==[(key,'normal',repeat) for key in expected_keys for repeat in range(3)]
   expected_extras=[]
   for dtype in ('fp32','bf16'):
    key=(dtype,4,248320,0);expected_extras.append((key,'nullable_score',0));expected_extras.extend((key,'graph',repeat) for repeat in range(3))
    expected_extras.append(((dtype,4,248320,10),'after_rejections',0))
   assert extras==expected_extras and observations_checked==1000 and graphs==['fp32','bf16']
   labels=['dimensions-0-248320','dimensions-129-248320','dimensions-4-0','dimensions-4--1','null-input','null-ids']
   assert rejections==[(dtype,label,1,1) for dtype in ('fp32','bf16') for label in labels]
   assert summaries==[{'kind':'summary','status':'parallel_batch_top1_correctness_passed','cases':330,'graphs':2,'rejected_calls':12,'performance_admitted':False}]
 else:assert not passed
 peak=None
 if base+'resources/resource-monitor.json' in manifest['files']:
  monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()]
  peak=max(s['used_bytes'] for s in samples)
  assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and monitor['child_exit_code']==events['gpu-ledger']==w['exit_code']
  assert monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
 elif passed:raise AssertionError('Missing GPU observation')
 archived=sha(z.read(methods+'verify_parallel_batch_top1.py'))
report={'status':'parallel_batch_top1_correctness_replayed' if passed else 'parallel_batch_top1_failure_preserved',
 'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'parent_status':w['parent_status'],
 'observed_unused_library_sha256':w['observed_unused_library_sha256'],'cases_checked':cases_checked,'raw_observations_checked':observations_checked,
 'raw_record_counts':counts,'raw_mismatches':mismatches,'truncated_failed_tail':truncated_tail,'correctness_sampled_peak_adapter_bytes':peak,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
 'replay_environment':{'python':platform.python_version(),'numpy':np.__version__},
 'production_integrated':False,'parent_model_qualification_promoted':False,'model_quality_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
