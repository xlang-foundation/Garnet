"""Teacher-forced precision/prefill diagnostic; no model promotion or timing."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
root=Path(sys.argv[1]);mode=sys.argv[2];out=Path(sys.argv[3]);out.mkdir(exist_ok=True)
assert mode in ('fp32-t4','bf16-t4','bf16-t32');bf16=mode!='fp32-t4';batch=4;steps=32
path=out/'report.json';assert not path.exists();sha=lambda b:hashlib.sha256(b).hexdigest()
checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes())
reference_path=root/'results/native-kv-delta-default-off-attempt1/native/report.json';reference_raw=reference_path.read_bytes();reference=json.loads(reference_raw)
old=next(c for c in reference['cases'] if c['group']==0 and c['repeat']==0)
assert old['keys']==['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy']*2
report={'status':'constructing','mode':mode,'batch':batch,'steps':steps,'register_delta_128':True,
 'library_sha256':sha((root/'build/bin/garnet.dll').read_bytes()),'reference_report_sha256':sha(reference_raw),
 'scope':'Diagnostic only. Same teacher-forced32token histories, JPEGs, mixed projections, register128, native NVFP4 head, host embedding and physicalB4. Compare FP32/T4 ordinary baseline, BF16/T4 manual ordinary prefill and BF16/T32 normal prefill. No MTP/timing or broad quality admission. Both BF16 modes use identical compiled T4/T32 programs; only prefill routing differs. Full-state arrays absent; raw pre-normalized/normalized hidden retained.'}
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save();decoder=None;vision=None
try:
 decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],batch,4096,32 if bf16 else 4,4,
  flat_concat=True,external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,
  host_embedding_lookup=True,joined_mlp_projections=True,shared_output_storage=True,register_delta_128=True,
  **({'bf16_kv_state':True} if bf16 else {}))
 decoder.io.enable_nvfp4_head(True);decoder.enable_cuda_graph(True);io=decoder.io
 pool=json.loads(io.memory_json());assert pool['head_precision']=='modelopt_nvfp4_w4a4' and pool['embedding_mode']=='host_mapped_bf16'
 assert pool['compressed_projection_pool_entries']==336 and pool['compressed_nvfp4_entries']==128 and pool['compressed_fp8_entries']==208
 report['pool']=pool;report['status']='preparing_inputs';save()
 vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
 rows=[];oracles=[]
 for index,key in enumerate(old['keys']):
  request=root/'results/native-mtp3-http'/(key+'-request.json');response=root/'results/native-mtp3-http'/(key+'-response.json')
  request_raw=request.read_bytes();response_raw=response.read_bytes();body=json.loads(request_raw);choice=json.loads(response_raw)['choices'][0]
  ids=choice['prompt_token_ids'];expected=choice['response_token_ids'];assert expected==old['generated_ids'][index] and len(expected)>=steps
  content=body['messages'][0]['content'];visual=None;start=-1
  if isinstance(content,list):
   item=next(x for x in content if x['type']=='image_url');data=base64.b64decode(item['image_url']['url'].split(',',1)[1],validate=True)
   image=root/'tmp'/('bf16-prefill-diagnostic-row-'+str(index)+'.jpg');image.write_bytes(data)
   prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
   visual=io.clone(vision.encode_patches(prepared[0]));start=ids.index(config['image_token_id']);assert ids.count(config['image_token_id'])==64
   positions,current=image_positions(len(ids),start,64,8)
  else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
  embeddings=io.embedding(ids);hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
  rows.append({'ids':ids,'positions':positions,'current':current,'expected':expected,'hidden':hidden,'visual':visual,'embeddings':embeddings})
  oracles.append({'key':key,'request_sha256':sha(request_raw),'response_sha256':sha(response_raw)})
 report['oracles']=oracles;lengths=[len(r['ids']) for r in rows];assert lengths==[41,104,41,104]
 count=max(lengths);zero=io.zeros([1,5120]);hidden=[io.stack_rows([r['hidden'][t] if t<len(r['hidden']) else zero for r in rows]) for t in range(count)]
 positions=[r['positions']+[[0,0,0]]*(count-len(r['ids'])) for r in rows]
 decoder.reset()
 if mode=='bf16-t4':
  # Use the already admitted T4 program directly; do not change any decoder,
  # graph, helper metadata, product admission or native source/precision flag.
  assert not decoder.prefill_mode and decoder.chunk_size==4
  for start in range(0,count,4):
   end=min(start+4,count)
   prediction=decoder.chunk_hidden(hidden[start:end],[[int(t<lengths[b]) for b in range(batch)] for t in range(start,end)],
    [p[start:end] for p in positions],predict=end==count)
 else:prediction=decoder.prefill_sequence(hidden,positions,lengths)
 assert decoder.slots==lengths and not decoder.prefill_mode and decoder.state_transaction is None
 width=decoder.programs[4].state_width
 buffers={n:io.zeros([batch,n]) for n in {min(1048576,width-i) for i in range(0,width,1048576)}}
 def state_hashes():
  result=[hashlib.sha256() for _ in range(batch)]
  for offset in range(0,width,1048576):
   n=min(1048576,width-offset);part=buffers[n];io.slice_into(decoder.state,offset,part)
   for b in range(batch):result[b].update(bytes(io.finite_f32_bytes(io.row(part,b))))
  return {'row_sha256':[h.hexdigest() for h in result],'slots':list(decoder.slots),'words_per_row':width}
 report['prefill_state']=state_hashes();report['predictions']=[];report['snapshots']=[];report['status']='teacher_forcing';save()
 current=[r['current'] for r in rows]
 for step in range(steps):
  normalized=decoder.head.run_device([decoder.last_hidden]);assert list(io.predict(normalized))==list(prediction)
  record={'step':step,'prediction':list(prediction),'teacher':[r['expected'][step] for r in rows],'raw':{}}
  for label,value in (('hidden',decoder.last_hidden),('normalized',normalized)):
   raw=bytes(io.finite_f32_bytes(value));assert len(raw)==batch*5120*4
   file='step-'+str(step)+'-'+label+'.f32';(out/file).write_bytes(raw);record['raw'][label]={'file':file,'sha256':sha(raw),'bytes':len(raw)}
  report['predictions'].append(list(prediction));report['snapshots'].append(record);save()
  if step+1<steps:
   prediction=decoder.step(record['teacher'],[1]*batch,[[v]*3 for v in current]);current=[v+1 for v in current]
 report['final_state']=state_hashes();report['memory']=decoder.memory_info()
 report['first_teacher_mismatch']=[next((i for i,p in enumerate(report['predictions']) if p[b]!=rows[b]['expected'][i]),None) for b in range(batch)]
 report['duplicate_pairs_exact']=all(p[:2]==p[2:] for p in report['predictions'])
 assert report['duplicate_pairs_exact']
 if mode=='fp32-t4':assert report['first_teacher_mismatch']==[None]*4,'FP32 reference control differs from retained teacher; do not interpret BF16 diagnostic'
 report['status']='teacher_forced_diagnostic_finished';save()
except BaseException as error:report.update(status='failed',error=repr(error));save();raise
finally:
 if vision is not None:vision.close()
 if decoder is not None:decoder.close()
print('BF16_PREFILL_DIAGNOSTIC_FINISHED',mode,flush=True)
