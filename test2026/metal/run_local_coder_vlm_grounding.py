"""Native staged image-to-code gate, one frozen image request per process.

XLang3 args: build, source, checkpoint folder, PNG, workload.json, result.json.
All model computation uses native Metal. Vision is closed before text loads;
the host embedding survives the handoff. No dummy image or text substitute.
"""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen38.vl_27b import model
folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)
with open(folder+'/preprocessor_config.json') as stream:processor=json.load(stream)
with open(sys.argv[5]) as stream:workload=json.load(stream)
if len(workload['requests'])!=1:raise ValueError('Grounding gate requires one frozen request')
tokens=workload['requests'][0]['token_ids'];limit=workload['max_output_tokens']
image_id=config['image_token_id'];indices=[i for i,token in enumerate(tokens) if token==image_id]
if len(indices)!=64 or indices!=list(range(indices[0],indices[0]+64)):
    raise ValueError('Grounding gate requires64 contiguous image tokens')
if type(limit) is not int or not 1<=limit<=1024 or len(tokens)+limit>4096:
    raise ValueError('Grounding output exceeds the native4K study profile')
vision=None;decoder=None;started=time.monotonic()
report=dict(status='running',scope='Staged native 256px image-to-code grounding; not performance or general VLM admission',
            backend=garnet.backend_info(),prompt_tokens=len(tokens),generated_token_ids=[],
            memory_before=json.loads(garnet.memory_stats_json()))
try:
    options=dict(patch_size=16,merge_size=2,temporal_patch_size=2,pixels=65536,
                 image_mean=processor['image_mean'],image_std=processor['image_std'])
    prepared=garnet.prepare_image(sys.argv[4],json.dumps(options))
    if [prepared[1],prepared[2]]!=[16,16]:raise ValueError('Unqualified grounding image grid')
    vision=model.create_metal_vision_study(folder+'/model.safetensors.index.json',config,16,16)
    embedding=vision.encode_patches(prepared[0])
    report['memory_vision_ready']=json.loads(garnet.memory_stats_json())
    vision.close();vision=None;prepared=None
    report['memory_after_vision_close']=json.loads(garnet.memory_stats_json())
    if report['memory_after_vision_close']['metal_allocated_bytes']>64*1024*1024:
        raise ValueError('Vision resources remained resident before text handoff')
    decoder=model.create_decoder(folder+'/model.safetensors.index.json',config,4096,linear_chunk=True)
    decoder.enable_prefill(8,'shared',True);decoder.reset()
    tokenizer=garnet.tokenizer();tokenizer.open(folder)
    stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
    hidden=[];positions=[];current=0;image_base=-1;image_index=0
    for token in tokens:
        if token==image_id:
            if image_index==0:image_base=current
            hidden.append(decoder.io.visual_token(embedding,image_index))
            positions.append([image_base,image_base+image_index//8,image_base+image_index%8])
            image_index+=1
            if image_index==64:current=image_base+8
        else:
            hidden.append(decoder.io.embedding([token]));positions.append([current,current,current]);current+=1
    report['image_embedding_tokens']=image_index
    report['memory_text_ready']=json.loads(garnet.memory_stats_json())
    predicted=decoder.prefill_sequence(hidden,[positions]);output=[];sampled=[];reason='length'
    report['ttft_seconds']=time.monotonic()-started
    for index in range(limit):
        token=predicted[0]
        if token<0 or token>=config['text_config']['vocab_size']:raise ValueError('Native token outside vocabulary')
        sampled.append(token)
        if token in stops:reason='stop';break
        output.append(token)
        if index+1<limit:
            predicted=decoder.step([token],active=[1],positions=[[current,current,current]]);current+=1
    report.update(status='completed',finish_reason=reason,text=tokenizer.decode(output),
                  generated_token_ids=sampled,generated_tokens=len(sampled),cached_prompt_tokens=0,
                  image_tokens=64,prompt_token_ids=tokens,total_seconds=time.monotonic()-started,
                  memory_completed=json.loads(garnet.memory_stats_json()))
    print('native_vlm_grounding_completed',len(sampled),reason,flush=True)
except:
    report['status']='failed'
    raise
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
    report['memory_after_close']=json.loads(garnet.memory_stats_json())
    with open(sys.argv[6],'w') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    garnet.shutdown()
