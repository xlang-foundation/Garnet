"""Replay the ordinary reference corpus; does not admit packed/MTP or speed."""
from pathlib import Path
import hashlib,json,math,struct,sys,zipfile
out=Path(sys.argv[1]);name='bf16-full-reference-corpus-attempt1';archive=out/(name+'-evidence.zip');sha=lambda b:hashlib.sha256(b).hexdigest();cases=[];raw_count=0
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));m=doc('evidence-manifest.json');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/'
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(m['files'])+1
 for n,h in m['files'].items():assert sha(z.read(n))==h
 for n,h in m['linked_archives'].items():assert sha((out/n).read_bytes())==h
 w=doc(base+'workflow.json');passed=w['exit_code']==0;assert w['cleanup_finished'] and w['library_unchanged'] and w['source_sha256']==w['source_after_sha256']
 for n,h in w['source_sha256'].items():assert sha(z.read('Garnet/'+n))==h
 raw=z.read(methods+'evidence/bf16-full-reference-corpus-preparation.json');p=json.loads(raw);assert sha(raw)==w['preparation_sha256'] and p['source_sha256']==w['source_sha256'] and p['linked_archives']==m['linked_archives']
 for n,h in p['methods'].items():assert sha(z.read(methods+n))==h
 for n,h in p['pinned_inputs'].items():assert sha(z.read(n))==h
 pr=z.read('results/bf16-same-precision-diagnostic-attempt1-verified.json');parent=json.loads(pr)
 assert sha(pr)==p['parent_proof_sha256']==w['parent_proof_sha256'] and parent['status']=='bf16_same_precision_diagnostic_replayed' and parent['library_sha256']==w['library_sha256']==p['library_sha256']
 assert parent['archive_sha256']==w['parent_archive_sha256']==m['linked_archives']['bf16-same-precision-diagnostic-attempt1-evidence.zip']
 for left,right in [('bf16-t4','reference-t4'),('bf16-t32','reference-t32')]:
  c=next(c for c in parent['comparisons'] if (c['left'],c['right'])==(left,right));assert c['all_raw_hidden_and_normalized_exact'] and c['all_predictions_exact'] and all(s['full_logical_state_exact'] for s in c['logical_state'].values())
 for ledger in ('cpu-ledger','ledger'):
  events=[json.loads(l) for l in z.read(base+ledger+'/commands.jsonl').splitlines()];assert len(events)==2 and events[0]['event']=='started' and events[-1]['event']=='finished' and events[-1]['exit_code']==(0 if ledger=='cpu-ledger' else w['exit_code'])
  if ledger=='cpu-ledger':log=z.read(base+ledger+'/commands.log');assert b'Ran 5 tests' in log and b'\nOK' in log
 if base+'native/report.json' in m['files']:
  r=doc(base+'native/report.json');assert r['batch']==4 and r['capacity']==4096 and r['prefill_chunk']==32 and r['decode_chunk']==4 and r['bf16_kv_float_reference'] and r['library_sha256']==w['library_sha256']
  assert r['model_config_sha256']==sha(z.read('models/Qwen3.8-27B-NVFP4/config.json'))
  lookup={}
  for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green'):
   prefix='results/native-mtp3-http/'+key;request=z.read(prefix+'-request.json');response=z.read(prefix+'-response.json');entry=next(e for e in r['oracles'] if e['key']==key)
   assert sha(request)==entry['request_sha256'] and sha(response)==entry['response_sha256'];lookup[key]=(json.loads(request),json.loads(response)['choices'][0])
  previous={}
  for c in r['cases']:
   assert c['keys']==(['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy']*2 if c['group']==0 else ['red','green']*2)
   assert c['prompt_lengths']==[len(lookup[k][1]['prompt_token_ids']) for k in c['keys']] and c['limits']==[lookup[k][0]['max_tokens'] for k in c['keys']]
   expected=[lookup[k][1]['response_token_ids'] for k in c['keys']];assert c['retained_fp32_ids']==expected and c['retained_fp32_ids_exact']==(c['generated_ids']==expected)
   assert c['generated_ids'][:2]==c['generated_ids'][2:] and all(0<len(ids)<=limit for ids,limit in zip(c['generated_ids'],c['limits']))
   for phase in ('prefill_state','final_state'):
    state=c[phase];assert state['physical_words_per_row']==state['logical_words_per_row']==173932544 and state['reference_bf16_roundtrip_checked']
    assert state['physical_row_sha256']==state['logical_row_sha256'] and len(state['components'])==128
    assert state['logical_row_sha256'][:2]==state['logical_row_sha256'][2:] and sum(s['logical_words'] for s in state['components'])==173932544
    for i,s in enumerate(state['components']):assert s['layer']==i//2 and s['role']==i%2+1 and s['kind']==('full_attention' if i//2%4==3 else 'linear_attention') and s['row_sha256'][:2]==s['row_sha256'][2:]
   assert c['prefill_state']['slots']==c['prompt_lengths'] and c['final_state']['slots']==[n+len(ids)-1 for n,ids in zip(c['prompt_lengths'],c['generated_ids'])]
   for label in ('hidden','normalized'):
    info=c['raw'][label];data=z.read(base+'native/'+info['file']);assert len(data)==info['bytes']==81920 and sha(data)==info['sha256'] and data[:40960]==data[40960:] and all(math.isfinite(v) for v in struct.unpack('<20480f',data));raw_count+=1
   if c['repeat']:
    old=previous[c['group']];assert c['generated_ids']==old['generated_ids'] and c['prefill_state']==old['prefill_state'] and c['final_state']==old['final_state']
    assert all(c['raw'][label]['sha256']==old['raw'][label]['sha256'] for label in ('hidden','normalized'))
   else:previous[c['group']]=c
   cases.append({'group':c['group'],'repeat':c['repeat'],'keys':c['keys'],'generated_token_counts':[len(ids) for ids in c['generated_ids']],'retained_fp32_ids_exact':c['retained_fp32_ids_exact']})
  if passed:assert r['status']=='bf16_full_reference_corpus_finished' and [(c['group'],c['repeat']) for c in r['cases']]==[(0,0),(0,1),(1,0),(1,1)] and raw_count==8 and w['status']=='bf16_full_reference_corpus_finished_needs_replay'
 else:assert not passed
 monitor=doc(base+'resources/resource-monitor.json');samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').splitlines()];peak=max(s['used_bytes'] for s in samples)
 assert monitor['child_exit_code']==w['exit_code'] and monitor['sampled_memory']['samples']==len(samples) and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6'
 archived=sha(z.read(methods+'verify_bf16_full_reference_corpus.py'))
r={'status':'bf16_full_reference_corpus_replayed' if passed else 'bf16_full_reference_corpus_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(m['files']),
 'library_sha256':w['library_sha256'],'cases':cases,'raw_final_hidden_and_normalized_arrays':raw_count,'correctness_sampled_peak_adapter_bytes':peak,
 'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'packed_mtp_qualified':False,'broad_quality_qualified':False,'performance_admitted':False,'scope':w['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
