"""Replay model vectors and logical-state witnesses, retaining every mismatch."""
from pathlib import Path
import hashlib,json,math,struct,sys,zipfile
out=Path(sys.argv[1]);name='bf16-same-precision-diagnostic-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
reports={};vectors={};resources={};comparisons=[];modes=['fp32-t4','bf16-t4','bf16-t32','reference-t4','reference-t32']
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0;assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']
 for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 raw=z.read(methods+'evidence/bf16-same-precision-diagnostic-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256']
 assert p['source_sha256']==w['source_sha256'] and p['linked_archives']==m['linked_archives']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 for n,h in p['pinned_inputs'].items():assert sha(z.read(n))==h
 proofraw=z.read('results/bf16-float-reference-model-build-attempt1-verified.json');proof=json.loads(proofraw)
 assert sha(proofraw)==w['build_proof_sha256'] and proof['status']=='bf16_float_reference_model_build_replayed' and proof['cpu_tests']==50 and proof['library_sha256']==w['library_sha256']==p['library_sha256']
 assert proof['archive_sha256']==m['linked_archives']['bf16-float-reference-model-build-attempt1-evidence.zip'] and proof['cpu_completion_archive_sha256']==m['linked_archives']['bf16-float-reference-cpu-completion-attempt1-evidence.zip']
 cpu=[json.loads(l) for l in z.read(base+'cpu-ledger/commands.jsonl').splitlines()];assert len(cpu)==2 and cpu[0]['event']=='started' and cpu[-1]['event']=='finished' and cpu[-1]['exit_code']==0
 cpulog=z.read(base+'cpu-ledger/commands.log');assert b'Ran 6 tests' in cpulog and b'\nOK' in cpulog
 oldraw=z.read('results/native-kv-delta-default-off-attempt1/native/report.json');old=json.loads(oldraw);teacher=next(c for c in old['cases'] if c['group']==0 and c['repeat']==0)['generated_ids']
 assert [c['mode'] for c in w['profiles']]==modes[:len(w['profiles'])]
 for entry in w['profiles']:
  mode=entry['mode'];prefix=base+mode+'/';events=[json.loads(l) for l in z.read(prefix+'ledger/commands.jsonl').splitlines()]
  assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==entry['exit_code']
  monitor=doc(prefix+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(prefix+'resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples)
  assert monitor['child_exit_code']==entry['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6';resources[mode]=peak
  if prefix+'native/report.json' not in m['files']:assert entry['exit_code']!=0;continue
  r=doc(prefix+'native/report.json');reports[mode]=r;vectors[mode]={}
  assert r['mode']==mode and r['batch']==4 and r['steps']==32 and r['library_sha256']==w['library_sha256'] and r['reference_report_sha256']==sha(oldraw) and r['register_delta_128']
  snapshots=r.get('snapshots',[])
  for i,s in enumerate(snapshots):
   assert s['step']==i and s['prediction']==r['predictions'][i] and s['teacher']==[t[i] for t in teacher]
   for label in ('hidden','normalized'):
    info=s['raw'][label];data=z.read(prefix+'native/'+info['file']);assert len(data)==info['bytes']==81920 and sha(data)==info['sha256']
    assert all(math.isfinite(x) for x in struct.unpack('<20480f',data)) and data[:40960]==data[40960:];vectors[mode][(i,label)]=data
  if entry['exit_code']==0:
   assert r['status']=='teacher_forced_diagnostic_finished' and len(snapshots)==32 and r['duplicate_pairs_exact']
   first=[next((i for i,s in enumerate(snapshots) if s['prediction'][b]!=teacher[b][i]),None) for b in range(4)];assert first==r['first_teacher_mismatch']
   if mode=='fp32-t4':assert first==[None]*4
   for key in ('prefill_state','final_state'):
    state=r[key];assert state['words_per_row']==(106823680 if mode.startswith('bf16-') else 173932544) and state['logical_words_per_row']==173932544
    assert state['reference_bf16_roundtrip_checked']==mode.startswith('reference-') and len(state['components'])==128
    assert state['row_sha256'][:2]==state['row_sha256'][2:] and state['logical_row_sha256'][:2]==state['logical_row_sha256'][2:]
    for i,c in enumerate(state['components']):
     assert c['layer']==i//2 and c['role']==i%2+1 and c['kind']==('full_attention' if c['layer']%4==3 else 'linear_attention') and c['row_sha256'][:2]==c['row_sha256'][2:]
    assert sum(c['logical_words'] for c in state['components'])==173932544
   assert r['prefill_state']['slots']==[41,104,41,104] and r['final_state']['slots']==[72,135,72,135]
 if passed:
  assert len(w['profiles'])==5 and all(c['exit_code']==0 for c in w['profiles']) and w['status']=='bf16_same_precision_diagnostic_finished_needs_replay' and w['default_off_parent_exact']
  with zipfile.ZipFile(out/'bf16-prefill-diagnostic-attempt1-evidence.zip') as parent:
   pr='results/bf16-prefill-diagnostic-attempt1/fp32-t4/native/';reference=json.loads(parent.read(pr+'report.json'))
   assert reports['fp32-t4']['predictions']==reference['predictions']
   for phase in ('prefill_state','final_state'):
    for key in ('row_sha256','slots','words_per_row'):assert reports['fp32-t4'][phase][key]==reference[phase][key]
   for (step,label),data in vectors['fp32-t4'].items():assert data==parent.read(pr+reference['snapshots'][step]['raw'][label]['file'])
  for left,right in [('bf16-t4','reference-t4'),('bf16-t32','reference-t32'),('reference-t4','reference-t32'),('bf16-t4','bf16-t32')]:
   states={};records=[]
   for phase in ('prefill_state','final_state'):
    a,b=reports[left][phase],reports[right][phase];different=[]
    for ca,cb in zip(a['components'],b['components']):
     assert (ca['layer'],ca['role'],ca['logical_words'])==(cb['layer'],cb['role'],cb['logical_words'])
     if ca['row_sha256']!=cb['row_sha256']:different.append({'layer':ca['layer'],'kind':ca['kind'],'role':ca['role'],'rows_equal':[x==y for x,y in zip(ca['row_sha256'],cb['row_sha256'])]})
    states[phase]={'full_logical_state_exact':a['logical_row_sha256']==b['logical_row_sha256'],'different_components':different}
   for step in range(32):
    for row in range(2):
     data=[vectors[v][(step,'normalized')] for v in (left,right)];a,b=[struct.unpack_from('<5120f',v,row*20480) for v in data]
     aa=math.fsum(x*x for x in a);bb=math.fsum(x*x for x in b);error=math.fsum((x-y)**2 for x,y in zip(a,b))
     records.append({'step':step,'row':row,'normalized_bit_exact':data[0][row*20480:(row+1)*20480]==data[1][row*20480:(row+1)*20480],
      'maximum_absolute_difference':max(abs(x-y) for x,y in zip(a,b)),'relative_l2_difference':math.sqrt(error/aa) if aa else None,
      'cosine':math.fsum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else None,'prediction_equal':reports[left]['predictions'][step][row]==reports[right]['predictions'][step][row]})
   comparisons.append({'left':left,'right':right,'all_raw_hidden_and_normalized_exact':vectors[left]==vectors[right],
    'all_predictions_exact':reports[left]['predictions']==reports[right]['predictions'],'logical_state':states,'points':records})
 archived=sha(z.read(methods+'verify_bf16_same_precision_diagnostic.py'))
r={'status':'bf16_same_precision_diagnostic_replayed' if passed else 'bf16_same_precision_diagnostic_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
   'library_sha256':w['library_sha256'],'first_teacher_mismatch':{k:v.get('first_teacher_mismatch') for k,v in reports.items()},'comparisons':comparisons,
   'correctness_sampled_peak_adapter_bytes':resources,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
   'model_qualified':False,'quality_promoted':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in r.items() if k!='comparisons'},indent=2))
