"""UNRUN packed BF16 private-padding, prefix, rollback and retirement controls."""
import base64,copy,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_batch_study_decoder,create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity
from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
from test2026.cuda.qwen38_5090.bf16_padded_state_gate import same_logical_row
root=Path(sys.argv[1]);batch=int(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(exist_ok=True)
mode=sys.argv[4];assert mode=='host';host_lookup=True
path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous batch qualification')
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_text(encoding='utf8'))
oracle=root/'results/native-mtp3-http'
keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green']
cases=[]
for key in keys:
 request=oracle/(key+'-request.json');response=oracle/(key+'-response.json')
 body=json.loads(request.read_text(encoding='utf8'));choice=json.loads(response.read_text(encoding='utf8'))['choices'][0]
 cases.append({'key':key,'body':body,'choice':choice,'oracle_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (request,response)}})
prior_path=root/'results/shared-mtp3-full-b2-deferred-attempt1/native/report.json'
prior_raw=prior_path.read_bytes();prior=json.loads(prior_raw)
assert prior['status']=='backend_smoke_completed'
prior_cases={(c['group'],c['repeat']):c for c in prior['cases']}
qualified_path=root/'results/native-kv-delta-default-off-attempt1/native/report.json'
qualified_raw=qualified_path.read_bytes();qualified=json.loads(qualified_raw);assert qualified['status']=='backend_smoke_completed'
qualified_cases={(c['group'],c['repeat']):c for c in qualified['cases']}
reference_root=root/'results/bf16-full-reference-corpus-attempt1/native'
reference_raw=(reference_root/'report.json').read_bytes();reference_report=json.loads(reference_raw)
assert reference_report['status']=='bf16_full_reference_corpus_finished' and reference_report['bf16_kv_float_reference'] and reference_report['batch']==4 and reference_report['prefill_chunk']==32 and reference_report['decode_chunk']==4
reference_cases={(c['group'],c['repeat']):c for c in reference_report['cases']};assert len(reference_cases)==4
report={'independent_reference_report_sha256':hashlib.sha256(reference_raw).hexdigest(),'joined_mlp_projections':True,'retained_unjoined_b4_report_sha256':hashlib.sha256(qualified_raw).hexdigest(),'host_embedding_lookup':host_lookup,'retained_state_report_sha256':hashlib.sha256(prior_raw).hexdigest(),'status':'starting','source':source_identity(),'batch':batch,'capacity':4096,'chunk':4,
 'garnet_dll_sha256':hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest(),
 'cases':[],'oracles':[{k:c[k] for k in ('key','oracle_sha256')} for c in cases],
 'mtp_depth':3,'draft_chunk':4,'deferred_state_commit':True,'draft_builder_workspace_bytes':256<<20,'kv_delta_state':False,'shared_output_storage':True,'bf16_kv_state':True,'prefill_chunk':32,'register_delta_128':True,'private_padding':True,'scope':'Prospective UNRUN physical B4/T32/T4 BF16 lifecycle gate. Seven groups repeated twice,1/2/3/4real rows, independent rowwise same-precision reference IDs/state/raw hidden+normalized; private dummy rows excluded explicitly. Mixed descriptive/color group tests retirement while other rows continue. Unequal0/1/2/4 accepted prefixes, rollback, pending ownership rejection and complete reset witnesses. Parent metadata/operator gates remain mandatory. Not broad quality, B8,4K, streaming/HTTP or speed admission.'}
path.write_text(json.dumps(report,indent=2));decoder=None;vision=None;speculative=None
try:
 assert batch == 4 and os.environ.get('GARNET_FP8_FULL_ATTN_M32_CANONICAL') == '1'
 from xModel.qwen38.vl_27b.model import validate_cuda_batch_study_profile
 from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
 validate_cuda_batch_study_profile(config,4096,batch)
 decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],batch,4096,32,4,
     flat_concat=True,external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,host_embedding_lookup=host_lookup,joined_mlp_projections=True,shared_output_storage=True,bf16_kv_state=True,register_delta_128=True)
 decoder.io.enable_nvfp4_head(True)
 pool=json.loads(decoder.io.memory_json()); report['external_compressed_pool']=pool
 assert pool['embedding_mode']==('host_mapped_bf16' if host_lookup else 'device_bf16')
 assert pool['embedding_table_bytes']==(0 if host_lookup else 2542796800)
 assert pool['host_embedding_mapped_tensor_bytes']==(2542796800 if host_lookup else 0)
 assert pool['compressed_projection_pool_entries']==336 and pool['compressed_nvfp4_entries']==128 and pool['compressed_fp8_entries']==208
 physical=pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']
 for program in decoder._all_programs():
  stats=json.loads(program.graph.stats_json())
  assert stats['native_nvfp4_projection_weights']==128 and stats['native_fp8_projection_weights']==208
  assert stats['external_bf16_weight_inputs']==0 and stats['external_compressed_weight_inputs']==336 and stats['external_compressed_weight_bytes']==physical
 report['external_compressed_weights']=True;path.write_text(json.dumps(report,indent=2))
 decoder.enable_cuda_graph(True);io=decoder.io
 from xModel.qwen38.vl_27b.model import mtp_shapes,validate_mtp_metadata
 from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
 validate_mtp_metadata(config,{name:json.loads(io.weight_info(name)) for name in mtp_shapes(config)})
 assert json.loads(io.memory_json())['head_precision']=='modelopt_nvfp4_w4a4'
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 speculative=CUDASpeculativeSession(decoder,str(checkpoint),stops,3,head_candidates=0,prefix_replay=True,draft_chunk_size=4,draft_builder_workspace_bytes=256<<20)
 width=decoder.programs[4].state_width
 def logical_state():
  assert decoder.state_transaction is None and not speculative.verifier.pending and not decoder.poisoned,'State hash requires healthy closed transaction'
  return logical_state_witness(io,decoder.state,decoder.programs[4].layout,batch,width,decoder.slots,packed_bf16=True)
 def state_hashes(label):
  witness=logical_state();last_hidden=bytes(io.finite_f32_bytes(decoder.last_hidden))
  normalized=bytes(io.finite_f32_bytes(decoder.head.run_device([decoder.last_hidden])))
  assert len(last_hidden)==len(normalized)==81920
  (out/(label+'-last-hidden.f32')).write_bytes(last_hidden)
  (out/(label+'-last-normalized.f32')).write_bytes(normalized)
  pair_bytes=2*config['text_config']['hidden_size']*4
  return {'state_row_sha256':witness['physical_row_sha256'],'logical_state':witness,
    'last_hidden_sha256':hashlib.sha256(last_hidden).hexdigest(),'last_normalized_sha256':hashlib.sha256(normalized).hexdigest(),
    'last_hidden_pair_sha256':[hashlib.sha256(last_hidden[offset:offset+pair_bytes]).hexdigest() for offset in range(0,len(last_hidden),pair_bytes)],
    'last_hidden_row_sha256':[hashlib.sha256(last_hidden[i*20480:(i+1)*20480]).hexdigest() for i in range(4)],
    'last_normalized_row_sha256':[hashlib.sha256(normalized[i*20480:(i+1)*20480]).hexdigest() for i in range(4)],
    'slots':list(decoder.slots),'state_words_per_row':width}
 def save():path.write_text(json.dumps(report,indent=2))
 def zero_sha(size):
  digest=hashlib.sha256();block=bytes(1048576)
  while size:
   n=min(size,len(block));digest.update(block[:n]);size-=n
  return digest.hexdigest()
 initial=state_hashes('constructor')
 assert initial['state_row_sha256']==[zero_sha(width*4)]*4 and initial['slots']==[0]*4
 assert initial['last_hidden_sha256']==initial['last_normalized_sha256']==zero_sha(81920)
 speculative.reset();empty=state_hashes('empty');assert empty==initial
 report['empty_state']=empty;report['prefix_controls']=[];save()
 def reference_for(key,repeat):
  matches=[(c,c['keys'].index(key)) for c in reference_report['cases'] if c['repeat']==repeat and key in c['keys']]
  assert len(matches)==1;return matches[0]
 def row_checks(state,phase,group,repeat):
  return [same_logical_row(state,row,reference_for(c['key'],repeat)[0][phase],reference_for(c['key'],repeat)[1]) for row,c in enumerate(group)]
 def vector_checks(label,group,repeat):
  checks={}
  for kind in ('hidden','normalized'):
   actual=(out/(label+'-last-'+kind+'.f32')).read_bytes();assert len(actual)==81920
   values=[]
   for row,c in enumerate(group):
    ref,index=reference_for(c['key'],repeat);entry=ref['raw'][kind];expected=(reference_root/entry['file']).read_bytes()
    assert len(expected)==81920 and hashlib.sha256(expected).hexdigest()==entry['sha256']
    values.append(actual[row*20480:(row+1)*20480]==expected[index*20480:(index+1)*20480])
   checks[kind]=values
  return checks
 def prefix_controls(hidden,positions,lengths,physical_rows,prediction):
  counts=[0,1,2,4];base=[r['current'] for r in physical_rows]
  tokens=[[reference_for(cases[row%2]['key'],0)[0]['generated_ids'][reference_for(cases[row%2]['key'],0)[1]][t] for row in range(4)] for t in range(4)]
  flags=[[1]*4 for _ in range(4)];pos=[[[p+t]*3 for t in range(4)] for p in base]
  before=state_hashes('prefix-before');baseline=list(prediction);current=list(base)
  for t in range(4):
   active=[int(t<n) for n in counts];next_ids=decoder.step(tokens[t],active,[[p]*3 for p in current])
   for row in range(4):
    if active[row]:baseline[row]=next_ids[row]
   current=[p+flag for p,flag in zip(current,active)]
  ordinary=state_hashes('prefix-ordinary')
  speculative.reset();speculative.prefill(hidden,positions,lengths,base)
  restored=state_hashes('prefix-restored');assert restored==before
  v=speculative.verifier;v.verify(tokens,flags,pos);rejected=[]
  for label,call in [
   ('parent-reset',lambda:decoder.reset()),
   ('parent-step',lambda:decoder.step(tokens[0],[1]*4,[[p]*3 for p in base])),
   ('nested-verify',lambda:v.verify(tokens,flags,pos)),
   ('prefix-overrun',lambda:v.commit_prefix([0,1,2,5])),
   ('prefix-bool',lambda:v.commit_prefix([True,1,2,4])),
   ('prefix-wrong-length',lambda:v.commit_prefix([0,1,2]))]:
   try:call()
   except ValueError as error:rejected.append({'label':label,'error':repr(error)})
   else:raise AssertionError('Pending ownership or prefix guard accepted '+label)
   assert v.pending and decoder.state_transaction is v
  v.rollback();rolled_back=state_hashes('prefix-rollback');assert rolled_back==before
  v.verify(tokens,flags,pos);committed=v.commit_prefix(counts);final=state_hashes('prefix-committed')
  record={'counts':counts,'tokens':tokens,'submitted_flags':flags,'positions':pos,'before':before,'restored':restored,'rollback':rolled_back,'ordinary':ordinary,'committed':final,
   'rejected':rejected,'predictions':committed,'ordinary_predictions':[baseline[row] if counts[row] else None for row in range(4)],
   'transaction_closed':not v.pending and decoder.state_transaction is None,'state_exact':final==ordinary}
  report['prefix_controls'].append(record);save()
  assert record['state_exact'] and record['transaction_closed'] and record['predictions']==record['ordinary_predictions']
  for kind in ('hidden','normalized'):
   assert (out/('prefix-ordinary-last-'+kind+'.f32')).read_bytes()==(out/('prefix-committed-last-'+kind+'.f32')).read_bytes()
  speculative.reset();assert state_hashes('prefix-reset')==empty
 groups=[cases[:1],cases[1:2],cases[:2],cases[:2]+cases[:1],cases[2:]+cases[2:3],cases[:2]*2,[cases[0],cases[2],cases[3]]]
 for group_index,group in enumerate(groups):
  for repeat in range(2):
   speculative.reset();stem='group-'+str(group_index)+'-repeat-'+str(repeat)
   reset=state_hashes(stem+'-reset');assert reset==empty;rows=[]
   for row,c in enumerate(group):
    ids=c['choice']['prompt_token_ids'];content=c['body']['messages'][0]['content'];visual=None;start=-1
    if isinstance(content,list):
     assert content[0]['type']=='image_url' and content[1]['type']=='text'
     data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)
     image=root/'tmp'/('batch-study-image-'+str(row)+'.jpg');image.write_bytes(data)
     prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
     visual=io.clone(vision.encode_patches(prepared[0]));start=ids.index(config['image_token_id']);assert ids.count(config['image_token_id'])==64
     positions,current=image_positions(len(ids),start,64,8)
    else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
    embeddings=io.embedding(ids)
    hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
    rows.append({'ids':ids,'positions':positions,'current':current,'hidden':hidden,'embeddings':embeddings,'visual':visual})
   real=len(rows);physical_rows=rows+[rows[0]]*(4-real);physical_keys=[c['key'] for c in group]+[group[0]['key']]*(4-real)
   count=max(len(r['ids']) for r in physical_rows);zero=io.zeros([1,5120])
   hidden=[io.stack_rows([r['hidden'][t] if t<len(r['ids']) else zero for r in physical_rows]) for t in range(count)]
   lengths=[len(r['ids']) for r in physical_rows];positions=[r['positions']+[[0,0,0]]*(count-len(r['ids'])) for r in physical_rows]
   limit=max(c['body']['max_tokens'] for c in group);assert limit in (32,128)
   prediction=decoder.prefill_sequence(hidden,positions,lengths)
   if group_index==5 and repeat==0:
    prefix_controls(hidden,positions,lengths,physical_rows,prediction)
    prediction=decoder.prefill_sequence(hidden,positions,lengths)
   ordinary_prefill=logical_state();ordinary=[[] for _ in rows];active=[int(row<real) for row in range(4)];current=[r['current'] for r in physical_rows]
   for step in range(limit):
    for row in range(real):
     if active[row]:
      ordinary[row].append(prediction[row])
      if prediction[row] in stops or len(ordinary[row])==limit:active[row]=0
    if not any(active):break
    prediction=decoder.step(prediction,active,[[p]*3 for p in current]);current=[p+flag for p,flag in zip(current,active)]
   ordinary_state=state_hashes(stem+'-ordinary')
   speculative.reset();speculative.prefill(hidden,positions,lengths,[r['current'] for r in physical_rows]);mtp_prefill=logical_state();callbacks=[]
   result=speculative.generate(limit,on_token=lambda row,token,sampled:callbacks.append([row,token,sampled]))
   all_generated=[list(v) for v in result['generated_token_ids']]
   for row,stop in enumerate(result['stop_token_ids']):
    if stop is not None:all_generated[row].append(stop)
   generated=all_generated[:real];mtp_state=state_hashes(stem+'-mtp')
   expected=[reference_for(c['key'],repeat)[0]['generated_ids'][reference_for(c['key'],repeat)[1]] for c in group]
   callback_rows=[[item[1] for item in callbacks if item[0]==row] for row in range(4)]
   callback_ordinals=all([item[2] for item in callbacks if item[0]==row]==list(range(1,len(all_generated[row])+1)) for row in range(4))
   checks={
    'ordinary_prefill':row_checks(ordinary_prefill,'prefill_state',group+[group[0]]*(4-real),repeat),
    'mtp_prefill':row_checks(mtp_prefill,'prefill_state',group+[group[0]]*(4-real),repeat),
    'ordinary_final':row_checks(ordinary_state['logical_state'],'final_state',group,repeat),
    'mtp_final':row_checks(mtp_state['logical_state'],'final_state',group,repeat)}
   vectors={mode:vector_checks(stem+'-'+mode,group,repeat) for mode in ('ordinary','mtp')}
   real_exact=all(ordinary_state[key][:real]==mtp_state[key][:real] for key in ('state_row_sha256','last_hidden_row_sha256','last_normalized_row_sha256','slots'))
   dummy_frozen=all(ordinary_state['state_row_sha256'][row]==ordinary_prefill['physical_row_sha256'][row] and ordinary_state['slots'][row]==ordinary_prefill['slots'][row] for row in range(real,4))
   dummy_duplicates=all(all(mtp_state[key][row]==mtp_state[key][0] for key in ('state_row_sha256','last_hidden_row_sha256','last_normalized_row_sha256','slots')) and all_generated[row]==all_generated[0] for row in range(real,4))
   iterations=result['speculative_metrics']['iterations'];retired=None
   if group_index==6:
    retired=all(len(generated[row])==2 and generated[row][-1] in stops for row in (1,2)) and len(generated[0])==128
    retired=retired and any(i['processed_target_counts'][0]>0 and i['processed_target_counts'][1:3]==[0,0] for i in iterations)
   record={'group':group_index,'repeat':repeat,'keys':[c['key'] for c in group],'physical_keys':physical_keys,'prompt_lengths':lengths,
    'real_rows':real,'padded_rows':4-real,'generation_limit':limit,'retained_request_limits':[c['body']['max_tokens'] for c in group],
    'generated_ids':generated,'ordinary_ids':ordinary,'expected_ids':expected,'all_physical_generated_ids':all_generated,
    'reset_state':reset,'ordinary_prefill_state':ordinary_prefill,'mtp_prefill_state':mtp_prefill,
    'ordinary_state':ordinary_state,'mtp_state':mtp_state,'independent_reference_rows_exact':checks,'independent_reference_vectors_exact':vectors,
    'real_physical_state_exact':real_exact,'ordinary_dummy_state_frozen':dummy_frozen,'mtp_dummy_duplicates_exact':dummy_duplicates,'callbacks':callbacks,'callbacks_exact':callback_rows==all_generated and callback_ordinals,
    'retired_rows_while_primary_continues':retired,'speculative_metrics':result['speculative_metrics']}
   report['cases'].append(record);save()
   assert generated==ordinary==expected and real_exact and record['callbacks_exact']
   assert all(all(rows) for rows in checks.values()) and all(all(rows) for mode in vectors.values() for rows in mode.values())
   assert retired is not False and dummy_frozen and dummy_duplicates
   if repeat:
    previous=report['cases'][-2]
    assert all(previous[key]==record[key] for key in ('generated_ids','ordinary_state','mtp_state','ordinary_prefill_state','mtp_prefill_state','callbacks'))
   print('BF16_PADDED_LIFECYCLE_CASE_PASS',group_index,repeat,flush=True)
 speculative.reset();report['final_reset_state']=state_hashes('final-reset');assert report['final_reset_state']==empty
 report['status']='bf16_padded_lifecycle_finished_needs_replay';save()
except BaseException as error:
 report.update(status='failed',error=repr(error));path.write_text(json.dumps(report,indent=2));raise
finally:
 if speculative is not None:speculative.close()
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
