"""Replay every real row, raw vector, prefix and reset without timing claims."""
from pathlib import Path
import hashlib,json,math,struct,sys,zipfile
from bf16_padded_state_gate import same_logical_row

out=Path(sys.argv[1]);name='bf16-padded-lifecycle-scope-fixed-attempt2'
sha=lambda b:hashlib.sha256(b).hexdigest()
archive=out/(name+'-evidence.zip');base='results/'+name+'/'
methods='Garnet/test2026/cuda/qwen38_5090/';checks=[];vectors_count=0
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json')
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 for n,h in manifest['linked_archives'].items():assert sha((out/n).read_bytes())==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0
 assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']
 for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 prepraw=z.read(methods+'evidence/bf16-padded-lifecycle-scope-fixed-preparation.json');p=json.loads(prepraw)
 assert sha(prepraw)==w['preparation_sha256'] and p['source_sha256']==w['source_sha256'] and p['linked_archives']==manifest['linked_archives']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 for n,h in p['pinned_inputs'].items():assert sha(z.read(n))==h
 for n in ('bf16_same_precision_gate.py','bf16_padded_state_gate.py'):
  assert sha(Path(__file__).with_name(n).read_bytes())==p['methods'][n]
 parent='bf16-same-precision-mtp-attempt1';proofraw=z.read('results/'+parent+'-verified.json');proof=json.loads(proofraw)
 assert sha(proofraw)==p['parent_proof_sha256']==w['parent_proof_sha256']
 assert proof['status']=='bf16_same_precision_mtp_corpus_replayed' and len(proof['cases'])==4 and proof['raw_hidden_and_normalized_arrays']==20
 assert not proof['historical_fp32_gate_reclassified'] and not proof['performance_admitted']
 assert proof['archive_sha256']==w['parent_archive_sha256']==manifest['linked_archives'][parent+'-evidence.zip']
 assert proof['library_sha256']==w['library_sha256']==p['library_sha256']
 fixture=z.read('results/bf16-kv-metadata-fixture.json');assert sha(fixture)==p['fixture_sha256']==w['fixture_sha256']
 recipe=doc(methods+'evidence/bf16-padded-lifecycle-probe-derivation.json')
 raw=z.read(methods+recipe['source']);assert sha(raw)==recipe['source_sha256']
 text=raw.decode().replace('\r\n','\n');text=text[:text.index(recipe['prefix_before_marker'])]
 for before,after in recipe['prefix_replacements']:
  assert text.count(before)==1;text=text.replace(before,after,1)
 result=(text+recipe['new_body']).encode()
 assert result==z.read(methods+'probe_bf16_padded_lifecycle.py') and sha(result)==recipe['result_sha256']
 fixed_recipe=doc(methods+'evidence/bf16-padded-lifecycle-scope-fixed-probe-derivation.json')
 assert fixed_recipe['source']=='probe_bf16_padded_lifecycle.py' and fixed_recipe['source_sha256']==sha(result)
 fixed=result.decode()
 for before,after in fixed_recipe['changes']:
  assert fixed.count(before)==1;fixed=fixed.replace(before,after,1)
 assert fixed.encode()==z.read(methods+'probe_bf16_padded_lifecycle_scope_fixed.py')
 assert sha(fixed.encode())==fixed_recipe['result_sha256']
 oldproof=doc('results/bf16-padded-lifecycle-attempt1-verified.json')
 assert oldproof['status']=='bf16_padded_lifecycle_failure_preserved'
 assert sha(z.read('results/bf16-padded-lifecycle-attempt1-verified.json'))==p['preserved_failure_proof_sha256']
 assert oldproof['archive_sha256']==manifest['linked_archives']['bf16-padded-lifecycle-attempt1-evidence.zip']
 with zipfile.ZipFile(out/(parent+'-evidence.zip')) as parent_zip:
  assert raw==parent_zip.read(methods+recipe['source'])
 reference_name='bf16-full-reference-corpus-attempt1';reference_base='results/'+reference_name+'/native/'
 reference=doc(reference_base+'report.json');assert reference['status']=='bf16_full_reference_corpus_finished'
 with zipfile.ZipFile(out/(reference_name+'-evidence.zip')) as ref_zip:
  for n in p['pinned_inputs']:
   if n.startswith(reference_base):assert z.read(n)==ref_zip.read(n)
 for ledger in ('cpu-ledger','ledger'):
  events=[json.loads(line) for line in z.read(base+ledger+'/commands.jsonl').splitlines()]
  assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished'
  assert events[-1]['exit_code']==(0 if ledger=='cpu-ledger' else w['exit_code'])
  if ledger=='cpu-ledger':
   log=z.read(base+ledger+'/commands.log');assert b'Ran 11 tests' in log and b'\nOK' in log

 def ref_for(key,repeat):
  found=[(c,c['keys'].index(key)) for c in reference['cases'] if c['repeat']==repeat and key in c['keys']]
  assert len(found)==1;return found[0]
 def raw_vector(label,kind):return z.read(base+'native/'+label+'-last-'+kind+'.f32')
 def raw_state(label,state):
  for kind in ('hidden','normalized'):
   data=raw_vector(label,kind)
   assert sha(data)==state['last_'+kind+'_sha256']
   assert [sha(data[i*20480:(i+1)*20480]) for i in range(4)]==state['last_'+kind+'_row_sha256']
  assert state['slots']==state['logical_state']['slots'] and state['state_row_sha256']==state['logical_state']['physical_row_sha256']
 for n in manifest['files']:
  if n.startswith(base+'native/') and n.endswith('.f32'):
   data=z.read(n);assert len(data)==81920 and all(math.isfinite(v) for v in struct.unpack('<20480f',data));vectors_count+=1
 if base+'native/report.json' in manifest['files']:
  r=doc(base+'native/report.json')
  assert r['garnet_dll_sha256']==w['library_sha256'] and r['independent_reference_report_sha256']==sha(z.read(reference_base+'report.json'))
  assert r['batch']==4 and r['capacity']==4096 and r['chunk']==4 and r['prefill_chunk']==32 and r['bf16_kv_state'] and not r['kv_delta_state'] and r['register_delta_128'] and r['private_padding']
  if passed:
   assert r['status']==w['status']=='bf16_padded_lifecycle_finished_needs_replay' and vectors_count==102
   assert [(c['group'],c['repeat']) for c in r['cases']]==[(g,k) for g in range(7) for k in range(2)]
   empty=r['empty_state'];raw_state('empty',empty)
   for label in ('constructor','empty','prefix-reset','final-reset'):
    assert all(raw_vector(label,kind)==bytes(81920) for kind in ('hidden','normalized'))
   expected_hash={}
   def zero_sha(size):
    if size not in expected_hash:
     h=hashlib.sha256();block=bytes(1048576);remaining=size
     while remaining:
      n=min(remaining,len(block));h.update(block[:n]);remaining-=n
     expected_hash[size]=h.hexdigest()
    return expected_hash[size]
   assert empty['state_words_per_row']==106823680 and empty['slots']==[0]*4 and empty['state_row_sha256']==[zero_sha(106823680*4)]*4
   assert empty['logical_state']['logical_row_sha256']==[zero_sha(173932544*4)]*4
   assert all(c['row_sha256']==[zero_sha(c['logical_words']*4)]*4 for c in empty['logical_state']['components'])
   assert r['final_reset_state']==empty and len(r['prefix_controls'])==1
   assert r['prefix_progress']['phase']=='passed'
   prefix=r['prefix_controls'][0];counts=prefix['counts'];assert counts==[0,1,2,4]
   assert prefix['before']==prefix['restored']==prefix['rollback'] and prefix['ordinary']==prefix['committed']
   assert prefix['state_exact'] and prefix['transaction_closed']
   assert [x['label'] for x in prefix['rejected']]==['parent-reset','parent-step','nested-verify','prefix-overrun','prefix-bool','prefix-wrong-length']
   expected_tokens=[[ref_for(('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy')[row%2],0)[0]['generated_ids'][row%2][t] for row in range(4)] for t in range(4)]
   assert prefix['tokens']==expected_tokens and prefix['submitted_flags']==[[1]*4]*4
   config=doc('models/Qwen3.8-27B-NVFP4/config.json');lengths=[];next_positions=[]
   for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy')*2:
    ids=doc('results/native-mtp3-http/'+key+'-response.json')['choices'][0]['prompt_token_ids']
    lengths.append(len(ids))
    if key=='s3-image-c1-r0-greedy':
     image_id=config['image_token_id'];start=ids.index(image_id)
     assert ids.count(image_id)==64 and ids[start:start+64]==[image_id]*64
     # Independent fixed8x8 merged-grid cursor:64cache slots advance
     # rotary THW coordinates by8, so cache length104 has next position48.
     next_positions.append(len(ids)-64+8)
    else:next_positions.append(len(ids))
   assert prefix['positions']==[[[p+t]*3 for t in range(4)] for p in next_positions]
   assert prefix['before']['slots']==lengths and prefix['committed']['slots']==[p+n for p,n in zip(lengths,counts)]
   expected_predictions=[ref_for(('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy')[row%2],0)[0]['generated_ids'][row%2][n] if n else None for row,n in enumerate(counts)]
   assert prefix['predictions']==prefix['ordinary_predictions']==expected_predictions
   for key in ('before','restored','rollback','ordinary','committed'):raw_state('prefix-'+key,prefix[key])
   for kind in ('hidden','normalized'):
    assert raw_vector('prefix-ordinary',kind)==raw_vector('prefix-committed',kind)
   t='s3-text-c1-r0-greedy';i='s3-image-c1-r0-greedy'
   groups=[[t],[i],[t,i],[t,i,t],['red','green','red'],[t,i,t,i],[t,'red','green']]
   for c in r['cases']:
    group=c['group'];repeat=c['repeat'];keys=groups[group];real=len(keys);stem='group-'+str(group)+'-repeat-'+str(repeat)
    assert c['keys']==keys and c['real_rows']==real and c['padded_rows']==4-real
    assert c['physical_keys']==keys+[keys[0]]*(4-real) and c['generation_limit']==(32 if group==4 else 128)
    assert c['retained_request_limits']==[32 if key in ('red','green') else 128 for key in keys]
    assert c['prompt_lengths']==[ref_for(key,repeat)[0]['prompt_lengths'][ref_for(key,repeat)[1]] for key in c['physical_keys']]
    expected=[ref_for(key,repeat)[0]['generated_ids'][ref_for(key,repeat)[1]] for key in keys]
    assert c['generated_ids']==c['ordinary_ids']==c['expected_ids']==expected and c['reset_state']==empty
    raw_state(stem+'-reset',c['reset_state'])
    computed={};vector_checks={}
    for mode in ('ordinary','mtp'):
     state=c[mode+'_state'];raw_state(stem+'-'+mode,state)
     for phase,ref_phase,selected in (('prefill','prefill_state',c['physical_keys']),('final','final_state',keys)):
      witness=c[mode+'_prefill_state'] if phase=='prefill' else state['logical_state']
      computed[mode+'_'+phase]=[same_logical_row(witness,row,ref_for(key,repeat)[0][ref_phase],ref_for(key,repeat)[1]) for row,key in enumerate(selected)]
     vector_checks[mode]={}
     for kind in ('hidden','normalized'):
      actual=raw_vector(stem+'-'+mode,kind);same=[]
      for row,key in enumerate(keys):
       ref,index=ref_for(key,repeat);entry=ref['raw'][kind];data=z.read(reference_base+entry['file']);assert sha(data)==entry['sha256']
       same.append(actual[row*20480:(row+1)*20480]==data[index*20480:(index+1)*20480])
      vector_checks[mode][kind]=same
    assert c['independent_reference_rows_exact']==computed and all(all(values) for values in computed.values())
    assert c['independent_reference_vectors_exact']==vector_checks and all(all(values) for mode in vector_checks.values() for values in mode.values())
    ordinary=c['ordinary_state'];mtp=c['mtp_state'];fields=('state_row_sha256','last_hidden_row_sha256','last_normalized_row_sha256','slots')
    assert c['real_physical_state_exact'] and all(ordinary[key][:real]==mtp[key][:real] for key in fields)
    assert c['ordinary_dummy_state_frozen'] and all(ordinary['state_row_sha256'][row]==c['ordinary_prefill_state']['physical_row_sha256'][row] and ordinary['slots'][row]==c['ordinary_prefill_state']['slots'][row] for row in range(real,4))
    assert c['mtp_dummy_duplicates_exact'] and all(all(mtp[key][row]==mtp[key][0] for key in fields) and c['all_physical_generated_ids'][row]==c['all_physical_generated_ids'][0] for row in range(real,4))
    callbacks=c['callbacks'];all_ids=c['all_physical_generated_ids'];assert all_ids[:real]==expected and c['callbacks_exact']
    assert all(type(v) is int for item in callbacks for v in item) and all(len(item)==3 and 0<=item[0]<4 for item in callbacks)
    for row in range(4):
     events=[item for item in callbacks if item[0]==row];assert [item[1] for item in events]==all_ids[row] and [item[2] for item in events]==list(range(1,len(all_ids[row])+1))
    retired=None
    if group==6:
     retired=len(expected[0])==128 and all(len(expected[row])==2 for row in (1,2)) and any(x['processed_target_counts'][0]>0 and x['processed_target_counts'][1:3]==[0,0] for x in c['speculative_metrics']['iterations'])
    assert c['retired_rows_while_primary_continues']==retired and retired is not False
    if repeat:
     previous=r['cases'][2*group]
     assert all(previous[key]==c[key] for key in ('generated_ids','ordinary_state','mtp_state','ordinary_prefill_state','mtp_prefill_state','callbacks'))
    checks.append({'group':group,'repeat':repeat,'real_rows':real,'ids_state_vectors_exact':True,'retired_rows_while_primary_continues':retired})
  else:
   checks=[{'group':c.get('group'),'repeat':c.get('repeat'),'available_case_fields':sorted(c)} for c in r.get('cases',[])]
 else:assert not passed
 monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+'resources/resource-vram.jsonl').splitlines()]
 peak=max(s['used_bytes'] for s in samples)
 assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
 assert monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
 archived_verifier=sha(z.read(methods+'verify_bf16_padded_lifecycle_scope_fixed.py'))
report={'status':'bf16_padded_lifecycle_replayed' if passed else 'bf16_padded_lifecycle_failure_preserved',
 'archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],
 'cases':checks,'raw_hidden_and_normalized_arrays':vectors_count,'correctness_sampled_peak_adapter_bytes':peak,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived_verifier,
 'broad_quality_qualified':False,'b8_qualified':False,'actual_4k_qualified':False,'streaming_http_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
