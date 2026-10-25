"""Unrun B4 T4 host-embedding MTP3 candidate; strict same-profile state and retained IDs."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_batch_study_decoder,create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.run_logged import source_identity
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
report={'joined_mlp_projections':True,'retained_unjoined_b4_report_sha256':hashlib.sha256(qualified_raw).hexdigest(),'host_embedding_lookup':host_lookup,'retained_state_report_sha256':hashlib.sha256(prior_raw).hexdigest(),'status':'starting','source':source_identity(),'batch':batch,'capacity':4096,'chunk':4,
 'garnet_dll_sha256':hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest(),
 'cases':[],'oracles':[{k:c[k] for k in ('key','oracle_sha256')} for c in cases],
 'mtp_depth':3,'draft_chunk':4,'deferred_state_commit':True,'draft_builder_workspace_bytes':256<<20,'kv_delta_state':True,'shared_output_storage':True,'private_padding':True,'scope':'Compact physicalB4/T4 with1/2/3/4real rows. Ordinary target leaves dummy generation rows inactive; MTP privately advances first-row duplicates. Strict real-row state hash/slot/rawhidden and retained IDs must agree, complete reset witnesses retained, and full-four controls retain same-DLL default-off parity. Full canonical arrays absent; no streaming/HTTP/actual4K/broadquality/speed admission.'}
path.write_text(json.dumps(report,indent=2));decoder=None;vision=None;speculative=None
try:
 assert batch == 4 and os.environ.get('GARNET_FP8_FULL_ATTN_M32_CANONICAL') == '1'
 from xModel.qwen38.vl_27b.model import validate_cuda_batch_study_profile
 from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
 validate_cuda_batch_study_profile(config,4096,batch)
 decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],batch,4096,4,4,
     flat_concat=True,external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,host_embedding_lookup=host_lookup,joined_mlp_projections=True,shared_output_storage=True,kv_delta_state=True)
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
 sizes={min(1048576,width-offset) for offset in range(0,width,1048576)}
 hash_buffers={size:io.zeros([batch,size]) for size in sizes}
 def state_hashes(label):
  assert decoder.state_transaction is None and not speculative.verifier.pending,'State hash requires closed transaction'
  digests=[hashlib.sha256() for _ in range(batch)]
  for offset in range(0,width,1048576):
   size=min(1048576,width-offset);part=hash_buffers[size]
   io.slice_into(decoder.state,offset,part)
   for row in range(batch):digests[row].update(bytes(io.finite_f32_bytes(io.row(part,row))))
  last_hidden=bytes(io.finite_f32_bytes(decoder.last_hidden))
  (out/(label+'-last-hidden.f32')).write_bytes(last_hidden)
  row_bytes=config['text_config']['hidden_size']*4
  pair_bytes=2*row_bytes
  return {'state_row_sha256':[h.hexdigest() for h in digests],
    'last_hidden_sha256':hashlib.sha256(last_hidden).hexdigest(),
    'last_hidden_pair_sha256':[hashlib.sha256(last_hidden[offset:offset+pair_bytes]).hexdigest() for offset in range(0,len(last_hidden),pair_bytes)],
    'last_hidden_row_sha256':[hashlib.sha256(last_hidden[offset:offset+row_bytes]).hexdigest() for offset in range(0,len(last_hidden),row_bytes)],
    'slots':list(decoder.slots),'state_words_per_row':width}
 # Final-write API guards are checked before any model request. State hashes
 # witness the complete canonical allocation; raw full state is not archived.
 negative_before=state_hashes('kv-negative-before')
 metadata=decoder.programs[4].delta_layout
 invalid_helper=garnet.cuda_kv_delta_commit();metadata_rejections=0
 def rejects(call,label):
  try:call()
  except Exception as error:return {'label':label,'error':repr(error)}
  raise AssertionError('Invalid final-write operation was accepted: '+label)
 for key in ('batch','capacity','chunk','hidden_width','canonical_state_width','output_width','linear_replay_width','payload_width'):
  invalid=json.loads(json.dumps(metadata));invalid[key]+=1
  rejects(lambda:invalid_helper.build(json.dumps(invalid)),'summary-'+key);metadata_rejections+=1
 for layer in range(64):
  invalid=json.loads(json.dumps(metadata));invalid['layers'][layer]['output_offset']+=1
  rejects(lambda:invalid_helper.build(json.dumps(invalid)),'output-offset-'+str(layer));metadata_rejections+=1
 for layer in range(64):
  key='value_offset' if layer%4==3 else 'replay_offset'
  invalid=json.loads(json.dumps(metadata));invalid['layers'][layer][key]+=1
  rejects(lambda:invalid_helper.build(json.dumps(invalid)),'state-offset-'+str(layer));metadata_rejections+=1
 assert metadata_rejections==136 and not json.loads(invalid_helper.stats_json())['ready']
 target=decoder.output_pool;linear=io.storage_view(target,[4,39714816]);alias_target=io.storage_view(decoder.state,[4,43030528]);alias_linear=io.storage_view(decoder.state,[4,39714816])
 rejected=[]
 for label,spans in [('length',[0]*7),('negative-start',[-1,1]+[0,0]*3),('fractional-start',[0.5,1]+[0,0]*3),('nonfinite-count',[0,float('nan')]+[0,0]*3),('too-many',[0,5]+[0,0]*3),('capacity-overrun',[4096,1]+[0,0]*3),('invalid-empty-start',[4097,0]+[0,0]*3)]:
  rejected.append(rejects(lambda:decoder.delta_commit.commit(target,target,decoder.state,spans),label))
 rejected.append(rejects(lambda:decoder.delta_commit.commit(alias_target,alias_target,decoder.state,[0,1]*4),'target-canonical-overlap'))
 rejected.append(rejects(lambda:decoder.delta_commit.commit(target,alias_linear,decoder.state,[0,1]*4),'linear-canonical-overlap'))
 wrong=io.storage_view(target,[4,43030527])
 rejected.append(rejects(lambda:decoder.delta_commit.commit(wrong,wrong,decoder.state,[0,1]*4),'target-width'))
 rejected.append(rejects(lambda:invalid_helper.commit(target,target,decoder.state,[0,0]*4),'closed-helper'))
 invalid_helper.close()
 decoder.delta_commit.commit(target,target,decoder.state,[4096,0]*4)
 decoder.delta_commit.commit(target,linear,decoder.state,[4096,0]*4)
 negative_after=state_hashes('kv-negative-after')
 assert negative_before==negative_after,'Rejected or zero-prefix commits changed canonical state'
 report['negative_commit_gate']={'status':'passed','metadata_rejections':metadata_rejections,'rejected_calls':rejected,'valid_zero_prefix_modes':2,'before':negative_before,'after':negative_after,'scope':'In-process full-state/last-hidden hash witnesses after malformed metadata/spans/overlap/width/closed-helper calls and two valid zero-prefix modes; raw full canonical arrays absent.'}
 report['kv_delta_contract']={'canonical_state_width':decoder.programs[4].state_width,'target_output_width':decoder.programs[4].output_width,'linear_replay_width':metadata['linear_replay_width'],'strategies':sorted(p.kv_overlay_strategy for p in decoder._all_programs()),'commit':json.loads(decoder.delta_commit.stats_json())}
 assert report['kv_delta_contract']['canonical_state_width']==173932544 and report['kv_delta_contract']['target_output_width']==43030528 and report['kv_delta_contract']['linear_replay_width']==39714816 and report['kv_delta_contract']['strategies']==[0,1]
 path.write_text(json.dumps(report,indent=2))
 del target,linear,alias_target,alias_linear,wrong
 groups=[cases[:1],cases[1:2],cases[:2],cases[:2]+cases[:1],cases[2:]+cases[2:3],cases[:2]*2]
 for group_index,group in enumerate(groups):
  for repeat in range(2):
   speculative.reset();reset_state=state_hashes('group-'+str(group_index)+'-repeat-'+str(repeat)+'-reset')
   assert reset_state==negative_before,'Private row state survived a complete request reset'
   rows=[]
   for row,c in enumerate(group):
    ids=c['choice']['prompt_token_ids'];content=c['body']['messages'][0]['content'];visual=None;start=-1
    if isinstance(content,list):
     item=next(x for x in content if x['type']=='image_url');data=base64.b64decode(item['image_url']['url'].split(',',1)[1],validate=True)
     image=root/'tmp'/('batch-study-image-'+str(row)+'.jpg');image.write_bytes(data)
     prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
     visual=io.clone(vision.encode_patches(prepared[0]));start=ids.index(config['image_token_id'])
     assert ids.count(config['image_token_id'])==64
     positions,current=image_positions(len(ids),start,64,8)
    else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
    embeddings=io.embedding(ids)
    hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
    rows.append({'ids':ids,'positions':positions,'current':current,'hidden':hidden,'embeddings':embeddings,'visual':visual})
   real_rows=len(rows);physical_rows=rows+[rows[0]]*(batch-real_rows)
   count=max(len(r['ids']) for r in physical_rows);zero=io.zeros([1,config['text_config']['hidden_size']])
   hidden=[io.stack_rows([r['hidden'][t] if t<len(r['hidden']) else zero for r in physical_rows]) for t in range(count)]
   lengths=[len(r['ids']) for r in physical_rows];positions=[r['positions']+[[0,0,0]]*(count-len(r['ids'])) for r in physical_rows]
   limits=[c['body']['max_tokens'] for c in group]
   assert len(set(limits))==1,'This bounded MTP probe requires equal per-row output limits'
   prediction=decoder.prefill_sequence(hidden,positions,lengths)
   ordinary=[[] for _ in rows];active=[int(row<real_rows) for row in range(batch)];current=[r['current'] for r in physical_rows]
   for step in range(limits[0]):
    for row in range(batch):
     if active[row]:
      ordinary[row].append(prediction[row])
      if prediction[row] in stops or len(ordinary[row])==limits[row]:active[row]=0
    if not any(active):break
    prediction=decoder.step(prediction,active,[[v]*3 for v in current])
    current=[v+flag for v,flag in zip(current,active)]
   ordinary_state=state_hashes('group-'+str(group_index)+'-repeat-'+str(repeat)+'-ordinary')
   speculative.reset()
   speculative.prefill(hidden,positions,lengths,[r['current'] for r in physical_rows])
   mtp_result=speculative.generate(limits[0])
   generated=[list(v) for v in mtp_result['generated_token_ids'][:real_rows]]
   for row,stop in enumerate(mtp_result['stop_token_ids'][:real_rows]):
    if stop is not None:generated[row].append(stop)
   mtp_state=state_hashes('group-'+str(group_index)+'-repeat-'+str(repeat)+'-mtp')
   expected=[c['choice']['response_token_ids'] for c in group]
   result={'group':group_index,'repeat':repeat,'keys':[c['key'] for c in group],'prompt_lengths':lengths,'limits':limits,
    'generated_ids':generated,'expected_ids':expected,'ordinary_ids':ordinary,'ordinary_state':ordinary_state,'mtp_state':mtp_state,'all_physical_rows_state_equal':ordinary_state==mtp_state,'real_rows':real_rows,'padded_rows':batch-real_rows,'reset_state':reset_state,'speculative_metrics':mtp_result['speculative_metrics'],'mtp_memory':speculative.memory_info(),'exact_b1_ids_match':generated==expected,'decoder_memory':decoder.memory_info()}
   def real_state_equal(a,b):
    return a['state_words_per_row']==b['state_words_per_row'] and all(a[key][:real_rows]==b[key][:real_rows] for key in ('state_row_sha256','last_hidden_row_sha256','slots'))
   result['real_rows_state_exact']=real_state_equal(ordinary_state,mtp_state)
   duplicates=[(i,j) for i in range(real_rows) for j in range(i) if group[i]['key']==group[j]['key']]
   result['duplicate_real_rows_exact']=all(state[key][i]==state[key][j] for state in (ordinary_state,mtp_state) for key in ('state_row_sha256','last_hidden_row_sha256','slots') for i,j in duplicates)
   result['same_DLL_default_off_full_control_exact']=None
   if real_rows==4:
    old=qualified_cases[(0,repeat)]
    strip=lambda state:{k:v for k,v in state.items() if k!='last_hidden_row_sha256'}
    result['same_DLL_default_off_full_control_exact']=generated==old['generated_ids'] and strip(ordinary_state)==old['ordinary_state'] and strip(mtp_state)==old['mtp_state']
   report['cases'].append(result);path.write_text(json.dumps(report,indent=2))
   assert result['same_DLL_default_off_full_control_exact'] is not False,'Full-four control differs from same-DLL default-off state or IDs'
   assert result['duplicate_real_rows_exact'],'Duplicate real rows diverged'
   assert generated==expected==ordinary,'Batched greedy output differs from B1 oracle; raw IDs saved'
   assert result['real_rows_state_exact'],'Real MTP target state differs from ordinary; private dummy state is separate'
   print('BATCH_BACKEND_FULL_IDS_PASS',batch,group_index,repeat,flush=True)
 report['status']='padded_mtp_state_passed_needs_replay';report['vision_memory']=json.loads(vision.graph.stats_json());path.write_text(json.dumps(report,indent=2))
except Exception as error:
 report['status']='failed';report['error']=repr(error);path.write_text(json.dumps(report,indent=2));raise
finally:
 if speculative is not None:speculative.close()
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
