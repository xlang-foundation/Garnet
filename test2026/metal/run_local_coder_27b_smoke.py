"""Native XLang3 bring-up only; run after all comparator workers exit.

Arguments: native-package directory, Garnet source, model directory, frozen
single-request workload.json, output JSON, optional output-token count (<=32).
This bounded 512-capacity run is not the 4K performance comparison.
"""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen38.vl_27b import model

folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[4]) as stream:workload=json.load(stream)
if len(workload['requests'])!=1:raise ValueError('Smoke requires one frozen request')
tokens=workload['requests'][0]['token_ids']
limit=int(sys.argv[6]) if len(sys.argv)>6 else 4
if limit<1 or limit>32 or len(tokens)+limit>512:
    raise ValueError('Bounded correctness smoke exceeds its initial profile')
decoder=None;tokenizer=None
report=dict(status='running',scope='Native 27B correctness bring-up; not matched performance admission',
    backend=garnet.backend_info(),capacity=512,prefill_chunk=8,batch=1,
    prompt_tokens=len(tokens),requested_output_tokens=limit,runs=[])
began=time.monotonic()
try:
    report['memory_before']=json.loads(garnet.memory_stats_json())
    decoder=model.create_decoder(folder+'/model.safetensors.index.json',config,512)
    decoder.enable_prefill(8,'shared',False)
    tokenizer=garnet.tokenizer();tokenizer.open(folder)
    report['initialization_seconds']=time.monotonic()-began
    report['memory_after_decoder']=json.loads(garnet.memory_stats_json())
    stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
    for request in range(2):
        decoder.reset();start=time.monotonic()
        hidden=[decoder.io.embedding([token]) for token in tokens]
        positions=[[[i,i,i] for i in range(len(tokens))]]
        predicted=decoder.prefill_sequence(hidden,positions)
        ids=[];durations=[]
        first=time.monotonic()
        for index in range(limit):
            token=predicted[0]
            if token<0 or token>=config['text_config']['vocab_size']:
                raise ValueError('Native prediction outside vocabulary')
            ids.append(token)
            if token in stops or index+1==limit:break
            step=time.monotonic();predicted=decoder.step(predicted)
            durations.append(time.monotonic()-step)
        report['runs'].append(dict(request=request,token_ids=ids,text=tokenizer.decode(ids),
            ttft_seconds=first-start,total_seconds=time.monotonic()-start,
            decode_step_seconds=durations,memory=json.loads(garnet.memory_stats_json())))
        with open(sys.argv[5],'w') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    if report['runs'][0]['token_ids']!=report['runs'][1]['token_ids']:
        raise ValueError('Reset changed deterministic native token sequence')
    report['status']='smoke_completed_not_full_qualification'
except:
    report['status']='failed'
    raise
finally:
    if decoder is not None:decoder.close()
    report['memory_after_close']=json.loads(garnet.memory_stats_json())
    with open(sys.argv[5],'w') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    garnet.shutdown()
