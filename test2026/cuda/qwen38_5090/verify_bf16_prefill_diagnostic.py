"""Replay teacher histories and raw-vector differences without quality promotion."""
from pathlib import Path
import hashlib,json,math,struct,sys,zipfile
out=Path(sys.argv[1]);name='bf16-prefill-diagnostic-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest()
reports={};vectors={};resources={};comparisons=[]
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0;assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']
 for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 raw=z.read(methods+'evidence/bf16-prefill-diagnostic-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 for n,h in p['pinned_inputs'].items():assert sha(z.read(n))==h
 for n,h in p['additional_source_sha256'].items():assert w['source_sha256'][n]==h
 parentname='bf16-model-b4-t32-attempt1';parent=doc('results/'+parentname+'-verified.json')
 assert sha(z.read('results/'+parentname+'-verified.json'))==w['parent_proof_sha256']
 assert parent['status']=='bf16_model_b4_t32_failure_preserved' and parent['ordinary_mtp_raw_hidden_exact'] and parent['library_sha256']==w['library_sha256']
 assert parent['archive_sha256']==w['parent_archive_sha256']==m['linked_archives'][parentname+'-evidence.zip']
 with zipfile.ZipFile(out/(parentname+'-evidence.zip')) as prior:
  parentreport=json.loads(prior.read('results/'+parentname+'/native/report.json'))
  parentworkflow=json.loads(prior.read('results/'+parentname+'/workflow.json'))
  assert w['source_sha256']=={**parentworkflow['source_sha256'],**p['additional_source_sha256']}
 oldraw=z.read('results/native-kv-delta-default-off-attempt1/native/report.json');old=json.loads(oldraw);ref=next(c for c in old['cases'] if c['group']==0 and c['repeat']==0)
 teacher=[r[:32] for r in ref['generated_ids']];expected_modes=['fp32-t4','bf16-t4','bf16-t32']
 assert [c['mode'] for c in w['profiles']]==expected_modes[:len(w['profiles'])]
 for entry in w['profiles']:
  mode=entry['mode'];prefix=base+mode+'/';events=[json.loads(l) for l in z.read(prefix+'ledger/commands.jsonl').splitlines()]
  assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==entry['exit_code']
  monitor=doc(prefix+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(prefix+'resources/resource-vram.jsonl').splitlines()]
  peak=max(s['used_bytes'] for s in samples);assert monitor['child_exit_code']==entry['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6';resources[mode]=peak
  if prefix+'native/report.json' not in m['files']:assert entry['exit_code']!=0;continue
  r=doc(prefix+'native/report.json');reports[mode]=r
  assert r['mode']==mode and r['batch']==4 and r['steps']==32 and r['library_sha256']==w['library_sha256'] and r['reference_report_sha256']==sha(oldraw) and r['register_delta_128']
  snapshots=r.get('snapshots',[]);vectors[mode]={}
  for i,s in enumerate(snapshots):
   assert s['step']==i and s['prediction']==r['predictions'][i] and s['teacher']==[t[i] for t in teacher]
   for label in ('hidden','normalized'):
    info=s['raw'][label];data=z.read(prefix+'native/'+info['file']);assert len(data)==info['bytes']==81920 and sha(data)==info['sha256']
    assert all(math.isfinite(x) for x in struct.unpack('<20480f',data)) and data[:40960]==data[40960:]
    vectors[mode][(i,label)]=data
  if entry['exit_code']==0:
   assert r['status']=='teacher_forced_diagnostic_finished' and len(snapshots)==32 and r['duplicate_pairs_exact']
   first=[next((i for i,s in enumerate(snapshots) if s['prediction'][b]!=teacher[b][i]),None) for b in range(4)];assert first==r['first_teacher_mismatch']
   if mode=='fp32-t4':assert first==[None]*4
   for key in ('prefill_state','final_state'):
    state=r[key];assert state['words_per_row']==(106823680 if mode!='fp32-t4' else 173932544) and len(state['row_sha256'])==4 and state['row_sha256'][:2]==state['row_sha256'][2:]
   assert r['prefill_state']['slots']==[41,104,41,104] and r['final_state']['slots']==[72,135,72,135]
 if passed:
  assert [c['mode'] for c in w['profiles']]==expected_modes and all(c['exit_code']==0 for c in w['profiles']) and w['status']=='bf16_prefill_diagnostic_finished_needs_replay'
  for left,right in (('fp32-t4','bf16-t4'),('bf16-t4','bf16-t32'),('fp32-t4','bf16-t32')):
   records=[]
   for step in range(32):
    for row in range(2):
     data=[vectors[mode][(step,'normalized')] for mode in (left,right)];a,b=[struct.unpack_from('<5120f',v,row*5120*4) for v in data]
     aa=math.fsum(x*x for x in a);bb=math.fsum(x*x for x in b);error=math.fsum((x-y)**2 for x,y in zip(a,b))
     records.append({'step':step,'row':row,'normalized_bit_exact':data[0][row*20480:(row+1)*20480]==data[1][row*20480:(row+1)*20480],
      'maximum_absolute_difference':max(abs(x-y) for x,y in zip(a,b)),'rms_difference':math.sqrt(error/5120),'relative_l2_difference':math.sqrt(error/aa) if aa else None,
      'cosine':math.fsum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb) if aa and bb else None,'prediction_equal':reports[left]['predictions'][step][row]==reports[right]['predictions'][step][row]})
   comparisons.append({'left':left,'right':right,'points':records,'prediction_agreements':sum(c['prediction_equal'] for c in records),'points_compared':len(records)})
 archived=sha(z.read(methods+'verify_bf16_prefill_diagnostic.py'))
result={'status':'bf16_prefill_diagnostic_replayed' if passed else 'bf16_prefill_diagnostic_failure_preserved','archive_sha256':sha(archive.read_bytes()),
 'files_verified':len(m['files']),'library_sha256':w['library_sha256'],'first_teacher_mismatch':{k:v.get('first_teacher_mismatch') for k,v in reports.items()},
 'comparisons':comparisons,'correctness_sampled_peak_adapter_bytes':resources,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,
 'quality_promoted':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
