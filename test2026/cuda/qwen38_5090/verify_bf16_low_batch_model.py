"""Replay closed B1 consistency evidence; never promote broad quality or speed."""
from pathlib import Path
import hashlib,json,sys,zipfile
from bf16_padded_state_gate import same_logical_row
sha=lambda b:hashlib.sha256(b).hexdigest()
out=Path(sys.argv[1]);name='bf16-low-batch-model-consistency-attempt1';archive=out/(name+'-evidence.zip');base='results/'+name+'/';methods='Garnet/test2026/cuda/qwen38_5090/';summaries=[];peaks={}
with zipfile.ZipFile(archive) as z:
 doc=lambda n:json.loads(z.read(n));manifest=doc('evidence-manifest.json');assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest['files'])+1
 for n,h in manifest['files'].items():assert sha(z.read(n))==h
 w=doc(base+'workflow.json');p=doc(methods+'evidence/bf16-low-batch-model-preparation.json')
 assert sha(z.read(methods+'evidence/bf16-low-batch-model-preparation.json'))==w['preparation_sha256']
 assert w['source_sha256']==w['source_after_sha256']==p['files'] and w['runtime_dependencies']==w['runtime_dependencies_after']==p['runtime_dependencies']
 assert w['library_unchanged'] and w['cleanup_finished'] and w['library_sha256']==p['runtime_dependencies']['garnet.dll'] and w['parents']==p['parents']
 for n,h in p['files'].items():assert sha(z.read(n))==h
 for n,row in p['parents'].items():
  raw=z.read('results/'+n+'-verified.json');q=json.loads(raw);assert sha(raw)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']
 passed=w['exit_code']==0
 parent_prefix='results/integrated-batch-top1-model-lifecycle-attempt1/enabled/';old=doc(parent_prefix+'report.json')
 ref=doc('results/bf16-full-reference-corpus-attempt1/native/report.json')
 def same(a,b):return a['logical']==b['logical'] and all(a['vectors'][k]['sha256']==b['vectors'][k]['sha256'] for k in ('hidden','normalized'))
 def validate_state(state,prefix,batch):
  logical=state['logical'];assert logical['physical_words_per_row']==106823680 and logical['logical_words_per_row']==173932544
  assert len(logical['physical_row_sha256'])==len(logical['logical_row_sha256'])==len(logical['slots'])==batch and len(logical['components'])==128
  assert all(type(n) is int and 0<=n<=4096 for n in logical['slots'])
  for entry in state['vectors'].values():
   assert Path(entry['file']).name==entry['file'];raw=z.read(prefix+entry['file']);assert len(raw)==entry['bytes']==batch*20480 and sha(raw)==entry['sha256']
  return state
 for entry in w['variants']:
  label=entry['label'];batch=entry['physical_batch'];prefix=base+label+'/';report_path=prefix+'report.json'
  if report_path not in manifest['files']:assert not passed;continue
  r=doc(report_path)
  assert r['library_sha256']==w['library_sha256'] and not r['quality_qualified'] and not r['performance_admitted']
  if not passed:continue
  assert entry['finished'] and entry['exit_code']==0 and (label,batch) in (('b4-smoke',4),('b1',1))
  assert r['physical_batch']==batch and r['capacity']==4096 and r['status']==('bf16_low_batch_b4_smoke_finished' if batch==4 else 'bf16_low_batch_model_consistency_finished')
  pool=r['pool_before'];assert pool['batch_top1_parallel_enabled'] and pool['head_precision']=='modelopt_nvfp4_w4a4' and (pool['compressed_projection_pool_entries'],pool['compressed_nvfp4_entries'],pool['compressed_fp8_entries'])==(336,128,208)
  assert r['helper']['ready'] and r['helper']['batch']==batch and r['helper']['chunk']==4 and r['prefill_helper']['chunk']==32
  empty=validate_state(r['empty_state'],prefix,batch);assert same(validate_state(r['final_reset'],prefix,batch),empty)
  keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy']+(['red','green'] if batch==1 else [])
  assert [(c['key'],c['repeat']) for c in r['cases']]==[(key,rep) for key in keys for rep in (0,1)]
  differences=[]
  for c in r['cases']:
   assert c['ordinary_mtp_exact'] and c['callbacks_exact'] and c['ordinary_ids']==c['generated_ids'] and len(c['ordinary_ids'])==batch
   for k in ('ordinary_state','mtp_state','prefill_state','mtp_prefill_state'):validate_state(c[k],prefix,batch)
   assert same(c['ordinary_state'],c['mtp_state']) and same(c['prefill_state'],c['mtp_prefill_state'])
   assert [[v[1] for v in c['callbacks'] if v[0]==row] for row in range(batch)]==c['generated_ids']
   assert all([v[2] for v in c['callbacks'] if v[0]==row]==list(range(1,len(c['generated_ids'][row])+1)) for row in range(batch))
   matches=[(v,v['keys'].index(c['key'])) for v in ref['cases'] if v['repeat']==c['repeat'] and c['key'] in v['keys']];assert len(matches)==1
   retained=matches[0][0]['generated_ids'][matches[0][1]];assert c['retained_b4_ids']==retained
   equal=all(ids==retained for ids in c['ordinary_ids']);assert c['retained_b4_ids_exact']==equal
   if not equal:differences.append({'key':c['key'],'repeat':c['repeat']})
   if batch==4:
    group=keys.index(c['key']);oldc=[v for v in old['cases'] if v['group']==group and v['repeat']==c['repeat']];assert len(oldc)==1;oldc=oldc[0]
    assert c['retained_b4_row0_state_and_raw_vectors_exact'] and c['ordinary_ids'][0]==oldc['ordinary_ids'][0]
    assert same_logical_row(c['ordinary_state']['logical'],0,oldc['ordinary_state']['logical_state'],0)
    assert c['ordinary_state']['logical']['physical_row_sha256'][0]==oldc['ordinary_state']['state_row_sha256'][0]
    for k in ('hidden','normalized'):assert z.read(prefix+c['ordinary_state']['vectors'][k]['file'])[:20480]==z.read(parent_prefix+'group-'+str(group)+'-repeat-'+str(c['repeat'])+'-ordinary-last-'+k+'.f32')[:20480]
   if c['repeat']:
    previous=r['cases'][r['cases'].index(c)-1];assert previous['generated_ids']==c['generated_ids'] and same(previous['ordinary_state'],c['ordinary_state']) and same(previous['mtp_state'],c['mtp_state'])
  if batch==1:
   assert [v['label'] for v in r['native_cross_dialect_rejections']]==['new-rejects-old','old-rejects-new']
   assert r['pending_guards_rejected']==['parent-reset','nested-verify','prefix-overrun','prefix-bool','prefix-wrong-length']
   assert [v['count'] for v in r['prefix_controls']]==[0,1,2,4]
   for v in r['prefix_controls']:
    assert v['exact'] and v['transaction_closed'];assert same(validate_state(v['ordinary'],prefix,batch),validate_state(v['committed'],prefix,batch))
  records=[json.loads(line) for line in z.read(base+label+'-ledger/commands.jsonl').splitlines()];assert len(records)==2 and records[0]['event']=='started' and records[1]['event']=='finished' and records[0]['id']==records[1]['id'] and records[1]['exit_code']==0
  monitor=doc(base+label+'-resources/resource-monitor.json');samples=[json.loads(line) for line in z.read(base+label+'-resources/resource-vram.jsonl').splitlines()]
  peak=max(s['used_bytes'] for s in samples);assert monitor['child_exit_code']==0 and monitor['gpu']['uuid']=='GPU-265ee66c-9b79-cd3b-0c63-dd6942ddb5e6' and monitor['sampled_memory']['sampled_peak_adapter_used_bytes']==peak
  peaks[label]=peak;summaries.append({'label':label,'physical_batch':batch,'cases':len(r['cases']),'ordinary_mtp_exact':True,'prefix_counts':[v['count'] for v in r['prefix_controls']],'retained_b4_token_difference_cases':differences,'retained_b4_row0_exact':batch==4,'broad_quality_qualified':False,'performance_admitted':False})
 if passed:assert w['status']=='bf16_low_batch_model_consistency_collected_needs_replay' and [v['label'] for v in summaries]==['b4-smoke','b1']
 archived=sha(z.read(methods+'verify_bf16_low_batch_model.py'))
report={'status':'bf16_low_batch_model_consistency_replayed' if passed else 'bf16_low_batch_model_consistency_failure_preserved','archive_sha256':sha(archive.read_bytes()),'files_verified':len(manifest['files']),'library_sha256':w['library_sha256'],'variants':summaries,'correctness_sampled_peak_adapter_bytes':peaks,'replay_verifier_sha256':sha(Path(__file__).read_bytes()),'archived_replay_verifier_sha256':archived,'broad_quality_qualified':False,'actual_4k_qualified':False,'streaming_http_qualified':False,'performance_admitted':False,'scope':p['scope']}
(out/(name+'-verified.json')).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2));raise SystemExit(0 if passed else 1)
