"""Real B1 model consistency and retained B4 smoke controls, never a timing run."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder,create_cuda_vision,prepare_cuda_image
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
from test2026.cuda.qwen38_5090.bf16_packed_state_gate import same_packed_bf16_row

root=Path(sys.argv[1]);batch=int(sys.argv[2]);out=Path(sys.argv[3]);assert batch in (1,4)
out.mkdir(exist_ok=True);path=out/'report.json';assert not path.exists()
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes())
assert os.environ.get('GARNET_PARALLEL_BATCH_TOP1')=='1' and os.environ.get('GARNET_FP8_FULL_ATTN_M32_CANONICAL')=='1'
report={'status':'starting','physical_batch':batch,'capacity':4096,'cases':[],'prefix_controls':[],'library_sha256':hashlib.sha256((root/'build/bin/garnet.dll').read_bytes()).hexdigest(),'quality_qualified':False,'performance_admitted':False,'scope':'B1 ordinary/MTP/prefix/reset consistency on four frozen text/image/color prompts, two repetitions; B4 smoke uses all duplicate rows internally and compares retained single-text/image row0 state and raw vectors exactly. B1-vs-B4 token differences are recorded, not silently accepted as quality equivalence. No complete legacy14case regression, actual4K, broad quality, streaming HTTP or performance qualification.'}
save=lambda:path.write_text(json.dumps(report,indent=2),encoding='utf8')
decoder=vision=session=None;save()
try:
 if batch==1:
  assert hasattr(garnet,'cuda_bf16_low_batch_kv_delta_commit')
  from xModel.qwen35.vl_0_8b.kv_bf16_low_batch_study_layout import build_bf16_low_batch_layout
  from xModel.qwen35.vl_0_8b.kv_bf16_study_layout import build_bf16_kv_study_layout
  report['native_cross_dialect_rejections']=[]
  for label,factory,layout in [('new-rejects-old',garnet.cuda_bf16_low_batch_kv_delta_commit,build_bf16_kv_study_layout(4,chunk=4,retain_linear_inputs=True)),('old-rejects-new',garnet.cuda_bf16_kv_delta_commit,build_bf16_low_batch_layout(chunk=4,retain_linear_inputs=True))]:
   helper=factory()
   try:
    try:helper.build(json.dumps(layout))
    except Exception as error:report['native_cross_dialect_rejections'].append({'label':label,'error':repr(error)})
    else:raise AssertionError('Native helper accepted another cache dialect '+label)
    assert not json.loads(helper.stats_json())['ready']
   finally:helper.close()
  assert len(report['native_cross_dialect_rejections'])==2;save()
  decoder=create_cuda_bf16_low_batch_study_decoder(str(checkpoint),config)
 else:
  decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],4,4096,32,4,flat_concat=True,external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,host_embedding_lookup=True,joined_mlp_projections=True,shared_output_storage=True,bf16_kv_state=True,register_delta_128=True)
  decoder.io.enable_nvfp4_head(True)
 decoder.enable_cuda_graph(True);io=decoder.io
 report['memory_before']=decoder.memory_info();report['pool_before']=json.loads(io.memory_json())
 assert report['pool_before']['batch_top1_parallel_enabled'] and report['pool_before']['head_precision']=='modelopt_nvfp4_w4a4'
 report['helper']=json.loads(decoder.delta_commit.stats_json());report['prefill_helper']=json.loads(decoder.bf16_prefill_commit.stats_json());save()
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
 session=CUDASpeculativeSession(decoder,str(checkpoint),stops,3,head_candidates=0,prefix_replay=True,draft_chunk_size=4,draft_builder_workspace_bytes=(128 if batch==1 else 256)<<20)
 width=decoder.programs[4].state_width
 def state(label):
  assert decoder.state_transaction is None and not session.verifier.pending and not decoder.poisoned
  logical=logical_state_witness(io,decoder.state,decoder.programs[4].layout,batch,width,decoder.slots,packed_bf16=True)
  vectors={}
  for kind,tensor in [('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))]:
   raw=bytes(io.finite_f32_bytes(tensor));assert len(raw)==batch*20480
   file=label+'-'+kind+'.f32';(out/file).write_bytes(raw);vectors[kind]={'file':file,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  return {'logical':logical,'vectors':vectors}
 def same(left,right):
  return left['logical']==right['logical'] and all(left['vectors'][k]['sha256']==right['vectors'][k]['sha256'] for k in ('hidden','normalized'))
 session.reset();empty=state('empty');report['empty_state']=empty;save()
 parent_root=root/'results/integrated-batch-top1-model-lifecycle-attempt1/enabled';parent=json.loads((parent_root/'report.json').read_bytes())
 assert parent['status']=='bf16_padded_lifecycle_finished_needs_replay'
 keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy']+(['red','green'] if batch==1 else [])
 reference=json.loads((root/'results/bf16-full-reference-corpus-attempt1/native/report.json').read_bytes())
 for index,key in enumerate(keys):
  oracle=root/'results/native-mtp3-http';request=oracle/(key+'-request.json');response=oracle/(key+'-response.json')
  body=json.loads(request.read_bytes());choice=json.loads(response.read_bytes())['choices'][0];ids=choice['prompt_token_ids'];content=body['messages'][0]['content'];visual=None;start=-1
  report.setdefault('inputs',{})[key]={'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),'response_sha256':hashlib.sha256(response.read_bytes()).hexdigest(),'prompt_token_ids':ids}
  if isinstance(content,list):
   data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True);image=out/(key+'.jpg');image.write_bytes(data)
   prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
   visual=io.clone(vision.encode_patches(prepared[0]));start=ids.index(config['image_token_id']);assert ids.count(config['image_token_id'])==64
   position,current=image_positions(len(ids),start,64,8)
  else:position=[[i]*3 for i in range(len(ids))];current=len(ids)
  embeddings=io.embedding(ids);hidden=[]
  for t in range(len(ids)):
   token=io.visual_token(visual,t-start) if visual is not None and start<=t<start+64 else io.row(embeddings,t)
   hidden.append(io.stack_rows([token]*batch))
  positions=[position for _ in range(batch)];lengths=[len(ids)]*batch;limit=body['max_tokens'];assert limit in (32,128)
  for repeat in range(2):
   stem=key+'-'+str(repeat);session.reset();reset=state(stem+'-reset');assert same(reset,empty)
   prediction=decoder.prefill_sequence(hidden,positions,lengths);prefill=state(stem+'-prefill');ordinary=[[] for _ in range(batch)];active=[1]*batch;slot=[current]*batch
   for step in range(limit):
    for row in range(batch):
     if active[row]:
      ordinary[row].append(prediction[row])
      if prediction[row] in stops or len(ordinary[row])==limit:active[row]=0
    if not any(active):break
    prediction=decoder.step(prediction,active,[[n]*3 for n in slot]);slot=[n+flag for n,flag in zip(slot,active)]
   ordinary_state=state(stem+'-ordinary')
   session.reset();session.prefill(hidden,positions,lengths,[current]*batch);mtp_prefill=state(stem+'-mtp-prefill');assert same(prefill,mtp_prefill)
   callbacks=[];result=session.generate(limit,on_token=lambda row,token,n:callbacks.append([row,token,n]));generated=[list(v) for v in result['generated_token_ids']]
   for row,stop in enumerate(result['stop_token_ids']):
    if stop is not None:generated[row].append(stop)
   mtp=state(stem+'-mtp');callback_ids=[[v[1] for v in callbacks if v[0]==row] for row in range(batch)]
   ref=[(c,c['keys'].index(key)) for c in reference['cases'] if c['repeat']==repeat and key in c['keys']];assert len(ref)==1
   retained=ref[0][0]['generated_ids'][ref[0][1]]
   record={'key':key,'repeat':repeat,'limit':limit,'ordinary_ids':ordinary,'generated_ids':generated,'ordinary_state':ordinary_state,'mtp_state':mtp,'prefill_state':prefill,'mtp_prefill_state':mtp_prefill,'callbacks':callbacks,'speculative_metrics':result['speculative_metrics'],'ordinary_mtp_exact':ordinary==generated and same(ordinary_state,mtp),'callbacks_exact':callback_ids==generated and all([v[2] for v in callbacks if v[0]==row]==list(range(1,len(generated[row])+1)) for row in range(batch)),'retained_b4_ids':retained,'retained_b4_ids_exact':all(v==retained for v in ordinary)}
   report['cases'].append(record);save();assert record['ordinary_mtp_exact'] and record['callbacks_exact']
   if batch==4:
    old=[c for c in parent['cases'] if c['group']==index and c['repeat']==repeat];assert len(old)==1;old=old[0]
    assert ordinary[0]==old['ordinary_ids'][0] and same_packed_bf16_row(ordinary_state['logical'],0,old['ordinary_state']['logical_state'],0)
    assert ordinary_state['logical']['physical_row_sha256'][0]==old['ordinary_state']['state_row_sha256'][0]
    for kind in ('hidden','normalized'):
     actual=(out/ordinary_state['vectors'][kind]['file']).read_bytes();expected=(parent_root/('group-'+str(index)+'-repeat-'+str(repeat)+'-ordinary-last-'+kind+'.f32')).read_bytes();assert actual[:20480]==expected[:20480]
    record['retained_b4_row0_state_and_raw_vectors_exact']=True;save()
   if repeat:
    old=report['cases'][-2];assert old['generated_ids']==generated and same(old['ordinary_state'],ordinary_state) and same(old['mtp_state'],mtp)
   if batch==1 and index==0 and repeat==0:
    assert len(ordinary[0])>=4
    tokens=[[ordinary[0][t]] for t in range(4)];flags=[[1] for _ in range(4)];pos=[[[current+t]*3 for t in range(4)]]
    for count in (0,1,2,4):
     session.reset();session.prefill(hidden,positions,lengths,[current]);before=state('prefix-'+str(count)+'-before')
     for t in range(count):decoder.step(tokens[t],[1],[[current+t]*3])
     target=state('prefix-'+str(count)+'-ordinary');session.reset();session.prefill(hidden,positions,lengths,[current]);assert same(state('prefix-'+str(count)+'-restored'),before)
     v=session.verifier;v.verify(tokens,flags,pos)
     if count==2:
      rejected=[]
      for label in ('parent-reset','nested-verify','prefix-overrun','prefix-bool','prefix-wrong-length'):
       try:
        if label=='parent-reset':decoder.reset()
        elif label=='nested-verify':v.verify(tokens,flags,pos)
        else:v.commit_prefix({'prefix-overrun':[5],'prefix-bool':[True],'prefix-wrong-length':[]}[label])
       except ValueError:rejected.append(label)
       else:raise AssertionError('Accepted invalid pending operation '+label)
      assert len(rejected)==5;report['pending_guards_rejected']=rejected
     v.rollback();assert same(state('prefix-'+str(count)+'-rollback'),before)
     v.verify(tokens,flags,pos);prediction=v.commit_prefix([count]);committed=state('prefix-'+str(count)+'-committed')
     record={'count':count,'ordinary':target,'committed':committed,'exact':same(target,committed),'prediction':prediction,'transaction_closed':decoder.state_transaction is None and not v.pending};report['prefix_controls'].append(record);save();assert record['exact'] and record['transaction_closed']
   print('B1_MODEL_CASE_PASS' if batch==1 else 'B4_REGRESSION_CASE_PASS',key,repeat,flush=True)
 session.reset();report['final_reset']=state('final-reset');assert same(report['final_reset'],empty)
 report['pool_after']=json.loads(io.memory_json());report['memory_after']=decoder.memory_info();report['status']='bf16_low_batch_model_consistency_finished' if batch==1 else 'bf16_low_batch_b4_smoke_finished';save()
except BaseException as error:report.update(status='failed',error=repr(error));save();raise
finally:
 if session is not None:session.close()
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
