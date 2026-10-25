"""Replay packed ordinary/MTP parity without converting FP32 failures to success."""
from pathlib import Path
import hashlib,json,math,struct,sys,zipfile
from bf16_same_precision_gate import same_logical_state
out=Path(sys.argv[1]);name='bf16-same-precision-mtp-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest();checks=[];raw_arrays=0
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0;assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']
 for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 raw=z.read(methods+'evidence/bf16-same-precision-mtp-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256'] and p['source_sha256']==w['source_sha256'] and p['linked_archives']==m['linked_archives']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 assert sha(Path(__file__).with_name('bf16_same_precision_gate.py').read_bytes())==p['methods']['bf16_same_precision_gate.py']
 for n,h in p['pinned_inputs'].items():assert sha(z.read(n))==h
 parentname='bf16-full-reference-corpus-attempt1';proofraw=z.read('results/'+parentname+'-verified.json');proof=json.loads(proofraw)
 assert sha(proofraw)==p['parent_proof_sha256']==w['parent_proof_sha256'] and proof['status']=='bf16_full_reference_corpus_replayed' and len(proof['cases'])==4 and proof['raw_final_hidden_and_normalized_arrays']==8
 assert proof['archive_sha256']==w['parent_archive_sha256']==m['linked_archives'][parentname+'-evidence.zip'] and proof['library_sha256']==w['library_sha256']==p['library_sha256']
 recipe=doc(methods+'evidence/bf16-same-precision-mtp-probe-derivation.json');raw=z.read(methods+recipe['source']);assert sha(raw)==recipe['source_sha256'];text=raw.decode().replace('\r\n','\n')
 a=text.index(' # Final-write API guards');b=text.index(' groups=[cases[:2]*2,cases[2:]*2]',a);control=text[a:b];assert sha(control.encode())==recipe['retained_metadata_negative_zero_prefix_block_sha256']
 for before,after in recipe['changes']:assert text.count(before)==1;text=text.replace(before,after,1)
 assert text.encode()==z.read(methods+'probe_bf16_same_precision_mtp.py') and sha(text.encode())==recipe['result_sha256'] and text.count(control)==1
 for ledger in ('cpu-ledger','ledger'):
  events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==(0 if ledger=='cpu-ledger' else w['exit_code'])
  if ledger=='cpu-ledger':log=z.read(base+ledger+'/commands.log');assert b'Ran 8 tests' in log and b'\nOK' in log
 fixture=z.read('results/bf16-kv-metadata-fixture.json');assert sha(fixture)==p['fixture_sha256']==w['fixture_sha256']
 references=doc('results/'+parentname+'/native/report.json');ref_cases={(c['group'],c['repeat']):c for c in references['cases']};assert references['status']=='bf16_full_reference_corpus_finished' and references['bf16_kv_float_reference']
 with zipfile.ZipFile(out/(parentname+'-evidence.zip')) as parent_zip:
  for n in p['pinned_inputs']:
   if n.startswith('results/'+parentname+'/native/'):assert z.read(n)==parent_zip.read(n)
 with zipfile.ZipFile(out/'bf16-model-b4-t32-attempt1-evidence.zip') as failed:
  assert failed.read(methods+recipe['source'])==z.read(methods+recipe['source'])
  historical=json.loads(failed.read('results/bf16-model-b4-t32-attempt1/native/report.json'));assert historical['status']=='failed' and historical['cases'][0]['ordinary_ids']!=historical['cases'][0]['expected_ids']
 old=doc('results/native-kv-delta-default-off-attempt1/native/report.json');old_cases={(c['group'],c['repeat']):c for c in old['cases']}
 if base+'native/report.json' in m['files']:
  r=doc(base+'native/report.json');assert r['garnet_dll_sha256']==w['library_sha256'] and r['independent_reference_report_sha256']==sha(z.read('results/'+parentname+'/native/report.json'))
  assert r['batch']==4 and r['capacity']==4096 and r['chunk']==4 and r['prefill_chunk']==32 and r['bf16_kv_state'] and not r['kv_delta_state'] and r['register_delta_128']
  for c in r['cases']:
   key=(c['group'],c['repeat']);ref=ref_cases[key];assert c['keys']==ref['keys'] and c['prompt_lengths']==ref['prompt_lengths'] and c['limits']==ref['limits']
   required=('ordinary_prefill_state','mtp_prefill_state','reference_prefill_exact','reference_final_exact','reference_raw_vectors_exact')
   if not passed and not all(field in c for field in required):
    checks.append({'group':key[0],'repeat':key[1],'status':'incomplete_case_preserved','available_case_fields':sorted(c),'producer_error':r.get('error')});continue
   ids=c['ordinary_ids']==c['generated_ids']==c['expected_ids']==ref['generated_ids'];physical=c['ordinary_state']==c['mtp_state']
   assert c['retained_fp32_reference_ids_exact']==(c['generated_ids']==old_cases[key]['generated_ids'])
   retained=[doc('results/native-mtp3-http/'+name+'-response.json')['choices'][0]['response_token_ids'] for name in c['keys']]
   assert c['retained_b1_ids_exact']==(c['generated_ids']==retained)
   pref=same_logical_state(c['ordinary_prefill_state'],ref['prefill_state']) and same_logical_state(c['mtp_prefill_state'],ref['prefill_state'])
   final=same_logical_state(c['ordinary_state']['logical_state'],ref['final_state']) and same_logical_state(c['mtp_state']['logical_state'],ref['final_state'])
   vectors={}
   for label in ('hidden','normalized'):
    expected=z.read('results/'+parentname+'/native/'+ref['raw'][label]['file']);assert len(expected)==81920 and sha(expected)==ref['raw'][label]['sha256']
    pair=[]
    for mode in ('ordinary','mtp'):
     filename='group-'+str(key[0])+'-repeat-'+str(key[1])+'-'+mode+'-last-'+label+'.f32';data=z.read(base+'native/'+filename)
     assert len(data)==81920 and data[:40960]==data[40960:] and all(math.isfinite(v) for v in struct.unpack('<20480f',data))
     assert sha(data)==c[mode+'_state']['last_'+label+'_sha256'];pair.append(data);raw_arrays+=1
    vectors[label]=pair[0]==pair[1]==expected
   assert c['reference_prefill_exact']==pref and c['reference_final_exact']==final and c['reference_raw_vectors_exact']==vectors
   checks.append({'group':key[0],'repeat':key[1],'ids_exact':ids,'packed_ordinary_mtp_physical_state_exact':physical,'independent_reference_prefill_exact':pref,'independent_reference_final_exact':final,'independent_reference_vectors_exact':vectors,'retained_fp32_reference_ids_exact':c['retained_fp32_reference_ids_exact']})
   if passed:assert ids and physical and pref and final and all(vectors.values()) and c['duplicate_pair_isolation_exact']
  if passed:
   assert r['status']=='backend_smoke_completed' and [(c['group'],c['repeat']) for c in r['cases']]==[(0,0),(0,1),(1,0),(1,1)] and raw_arrays==16 and w['status']=='bf16_same_precision_mtp_finished_needs_replay'
   assert r['runtime_metadata_gate']=={'valid_profiles':4,'rebuild_rejections':4,'invalid_layouts':8856,'fixture_sha256':p['fixture_sha256']}
   negative=r['negative_commit_gate'];assert negative['status']=='passed' and negative['before']==negative['after'] and negative['valid_zero_prefix_modes']==3 and len(negative['rejected_calls'])==16
   for phase in ('before','after'):
    prefix='kv-negative-'+phase
    for label in ('hidden','normalized'):
     data=z.read(base+'native/'+prefix+'-last-'+label+'.f32');assert len(data)==81920 and sha(data)==negative[phase]['last_'+label+'_sha256'];raw_arrays+=1
   contract=r['kv_delta_contract'];assert contract['canonical_state_width']==106823680 and contract['target_output_width']==43030528 and contract['linear_replay_width']==39714816 and contract['strategies']==[1,1] and contract['commit']['ready'] and not contract['commit']['poisoned'] and contract['commit']['commits']==2
 else:assert not passed
 monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples)
 assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
 archived=sha(z.read(methods+'verify_bf16_same_precision_mtp.py'))
report={'status':'bf16_same_precision_mtp_corpus_replayed' if passed else 'bf16_same_precision_mtp_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
 'library_sha256':w['library_sha256'],'cases':checks,'raw_hidden_and_normalized_arrays':raw_arrays,'correctness_sampled_peak_adapter_bytes':peak,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'historical_fp32_gate_reclassified':False,'broad_quality_qualified':False,'b8_qualified':False,'actual_4k_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
