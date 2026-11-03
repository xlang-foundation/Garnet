"""Single active native Metal request, exact rendered prompt, real token events.

XLang3 arguments: native build, source root, checkpoint folder. This study
worker deliberately provides no continuous batching or prefix cache claim.
"""
import sys,json,os,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen38.vl_27b import model

def emit(value):
    raw=(json.dumps(value,ensure_ascii=True,separators=(',',':'))+'\n').encode()
    while raw:
        count=os.write(1,raw)
        if count<1:raise OSError('Native study event pipe closed')
        raw=raw[count:]

def requests():
    pending=b''
    while True:
        block=os.read(0,65536)
        if not block:
            if pending:raise ValueError('Incomplete native request frame')
            return
        pending+=block
        while b'\n' in pending:
            line,pending=pending.split(b'\n',1)
            if len(line)>2*1024*1024:raise ValueError('Native request frame too large')
            yield json.loads(line)
        if len(pending)>2*1024*1024:raise ValueError('Native request frame too large')

folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)
decoder=None
try:
    decoder=model.create_decoder(folder+'/model.safetensors.index.json',config,4096,linear_chunk=True)
    decoder.enable_prefill(8,'shared',True)
    tokenizer=garnet.tokenizer();tokenizer.open(folder)
    stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
    emit(dict(event='ready',backend=garnet.backend_info(),active_requests=1,capacity=4096,
              prefill_chunk=8,prefix_cache=False,multimodal=False,
              memory=json.loads(garnet.memory_stats_json())))
    for request in requests():
        key=request['id'];body=request['body']
        try:
            ids=list(tokenizer.encode(body['rendered_prompt']));limit=body['max_tokens']
            if type(limit) is not int or not 2<=limit<=1024 or not ids or len(ids)>3072 or len(ids)+limit>4096:
                raise ValueError('Native request exceeds the 4K study profile')
            decoder.reset()
            hidden=[decoder.io.embedding([token]) for token in ids]
            positions=[[[i,i,i] for i in range(len(ids))]]
            predicted=decoder.prefill_sequence(hidden,positions)
            generated=[];text_ids=[];reason='length';began=time.monotonic()
            for index in range(limit):
                token=predicted[0]
                if token<0 or token>=config['text_config']['vocab_size']:
                    raise ValueError('Native token outside vocabulary')
                generated.append(token)
                if token not in stops:text_ids.append(token)
                text=tokenizer.decode(text_ids).encode('utf8').decode('utf8','replace')
                emit(dict(event='token',id=key,token_id=token,sampled_tokens=len(generated),
                          prompt_tokens=len(ids),text=text))
                if token in stops:reason='stop';break
                if index+1<limit:
                    position=len(ids)+index
                    predicted=decoder.step([token],active=[1],positions=[[position,position,position]])
            emit(dict(event='completed',id=key,generated_ids=generated,prompt_ids=ids,text=text,
                      finish_reason=reason,generation_mode='native_metal_greedy_single_active',
                      decode_window_seconds=time.monotonic()-began,
                      memory=json.loads(garnet.memory_stats_json())))
        except Exception as error:
            emit(dict(event='error',id=key,error=str(error)))
            # A native failure may leave poisoned state. Never serve another
            # request under an apparently healthy profile after such a failure.
            raise
finally:
    if decoder is not None:decoder.close()
    garnet.shutdown()
