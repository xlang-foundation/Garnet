"""Native Metal continuous-batch text study worker.

XLang args: native build, Garnet source, checkpoint folder, frozen profile JSON.
CPU code supplies tokenization/embeddings and transports real native events;
row allocation, scheduling, prefix matching and hybrid state live in native code.
Prefill prompts share a chunk graph. Profiles explicitly choose whole-prompt
or bounded chunk scheduling; an optional mixed forward remains experimental.
"""
import sys,json,os,time,threading,queue
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen38.vl_27b.metal_batch_study import create_decoder

def emit(value):
    raw=(json.dumps(value,ensure_ascii=True,separators=(',',':'))+'\n').encode()
    while raw:
        count=os.write(1,raw)
        if count<1:raise OSError('Native batch event pipe closed')
        raw=raw[count:]

def f32(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)

def main():
    folder=sys.argv[3]
    with open(sys.argv[4]) as stream:profile=json.load(stream)
    with open(folder+'/config.json') as stream:config=json.load(stream)
    batch=profile['batch'];capacity=profile['capacity'];chunk=profile['prefill_chunk'];prefix=profile.get('prefix_cache',False)
    if type(prefix) is not bool:raise ValueError('Explicit boolean prefix policy required')
    prefill_schedule=profile.get('prefill_schedule','whole')
    if prefill_schedule not in ('whole','chunked'):raise ValueError('Explicit supported prefill scheduling mode required')
    chunked=prefill_schedule=='chunked'
    mixed=profile.get('mixed_forward',False)
    if type(mixed) is not bool or (mixed and (not chunked or prefix)):
        raise ValueError('Mixed forward requires explicit uncached chunked scheduling')
    if mixed and 'packed_token_plan' not in garnet.backend_info():
        raise ValueError('Mixed forward requires the native packed plan extension')
    indexed=profile.get('indexed_projections',False)
    if type(indexed) is not bool or indexed!=('indexed_token_projections' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit indexed projection policy')
    retire_packs=profile.get('retire_nonresident_packs',False)
    if type(retire_packs) is not bool or retire_packs!=('retired_nonresident_packs' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit nonresident pack policy')
    large=profile.get('large_prefill',False)
    if type(large) is not bool or large!=('qwen38_large_prefill' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit large prefill policy')
    reuse64=profile.get('require_reuse64',False)
    if type(reuse64) is not bool or reuse64!=('mpp_affine_reuse64_optional' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit cooperative64 projection policy')
    if reuse64 and (profile.get('reuse64_min_outputs')!=5120 or 'mpp_affine_reuse64_min_outputs=5120' not in garnet.backend_info()):
        raise ValueError('Frozen cooperative64 profile requires its measured large-output threshold')
    reuse64_relaxed=profile.get('reuse64_relaxed',False)
    if type(reuse64_relaxed) is not bool or reuse64_relaxed!=('mpp_affine_reuse64_relaxed=1' in garnet.backend_info()):
        raise ValueError('Frozen cooperative64 profile requires its explicit tensor precision mode')
    active_decode=profile.get('active_decode_projections',False)
    kernel_profiles=profile.get('bounded_affine_kernel_profiles',False)
    if type(kernel_profiles) is not bool or kernel_profiles!=('bounded_affine_kernel_profiles' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit bounded kernel compilation policy')
    if kernel_profiles:
        index_bits=profile.get('affine_profile_index_bits',64)
        if type(index_bits) is not int or index_bits not in (32,64) or 'affine_profile_index_bits='+str(index_bits) not in garnet.backend_info():
            raise ValueError('Frozen compiled profile requires its explicit native index width')
        thread_limit=profile.get('affine_profile_max_threads',0)
        if type(thread_limit) is not int or thread_limit not in (0,128) or 'affine_profile_max_threads='+str(thread_limit) not in garnet.backend_info():
            raise ValueError('Frozen compiled profile requires its explicit compiler thread bound')
        close_limit=profile.get('compiled_engine_close_bytes_limit')
        if type(close_limit) is not int or not 0<close_limit<=4*1024*1024:raise ValueError('Compiled engine requires its frozen bounded post-close driver allocation limit')
        for key in ('expected_body_affine_projections','expected_head_affine_projections'):
            if type(profile.get(key)) is not int or profile[key]<=0:raise ValueError('Compiled kernel profile requires explicit projection coverage')
    retire_prefill_inputs=profile.get('retire_prefill_hidden_inputs',False)
    if type(retire_prefill_inputs) is not bool or retire_prefill_inputs!=('retired_prefill_hidden_inputs' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit prefill hidden input retirement policy')
    if type(active_decode) is not bool or active_decode!=('active_decode_projections' in garnet.backend_info()):
        raise ValueError('Frozen profile requires its explicit active FP32 decode projection policy')
    if active_decode:
        for key in ('expected_body_indexed_decode_projections','expected_head_indexed_decode_projections'):
            if type(profile.get(key)) is not int or profile[key]<=0:raise ValueError('Active decode requires explicit expected projection coverage')
    if 'require_attention_prefix' in profile:
        attention_prefix=profile['require_attention_prefix']
        if type(attention_prefix) is not bool or attention_prefix!=('attention_prefix' in garnet.backend_info()):
            raise ValueError('Frozen profile requires its explicit attention slot lookup policy')
    if 'require_lifetime_reuse' in profile:
        lifetime=profile['require_lifetime_reuse']
        if type(lifetime) is not bool or lifetime!=('prefill_lifetime_reuse' in garnet.backend_info()):
            raise ValueError('Frozen profile requires its explicit scratch lifetime policy')
    if chunked and (prefix or 'chunked_prefill_scheduler' not in garnet.backend_info()):
        raise ValueError('Chunked prefill study requires native chunk scheduling and an uncached profile')
    if not profile.get('profile_id'):raise ValueError('Frozen profile identity required')
    if 'row_final_hidden' not in garnet.backend_info():raise ValueError('Native per-row final hidden extension required')
    if profile.get('require_mpp',False):
        tag={'float32':'mpp_affine_fp32_optional','bfloat16':'mpp_affine_bf16_optional'}.get(profile.get('mpp_operand_dtype','float32'))
        if tag is None or tag not in garnet.backend_info():raise ValueError('Frozen profile requires its explicit native MPP operand type')
        tile=profile.get('mpp_tile_k',32)
        if type(tile) is not int or tile not in (32,64,128) or 'mpp_affine_tile_k='+str(tile) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP reduction tile')
        relaxed=profile.get('mpp_relaxed',False)
        if type(relaxed) is not bool or 'mpp_affine_relaxed='+str(int(relaxed)) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP tensor precision mode')
        layout=profile.get('mpp_n_major',False)
        if type(layout) is not bool or 'mpp_affine_n_major='+str(int(layout)) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP weight layout')
        registers=profile.get('mpp_registers',False)
        if type(registers) is not bool or 'mpp_affine_registers='+str(int(registers)) not in garnet.backend_info():
            raise ValueError('Frozen profile requires its explicit MPP operand storage')
        tensor_k=16 if registers else tile
        if 'mpp_affine_tensor_k='+str(tensor_k) not in garnet.backend_info():raise ValueError('Frozen profile requires its explicit MPP tensor reduction extent')
    width=config['text_config']['hidden_size'];vocab=config['text_config']['vocab_size']
    incoming=queue.Queue(maxsize=16);shutdown=threading.Event();input_failure=[]
    decoder=None;policy=None;jobs={};keys=set()
    def read():
        pending=b''
        try:
            while not shutdown.is_set():
                block=os.read(0,65536)
                if not block:
                    if pending:raise ValueError('Incomplete native batch request frame')
                    return
                pending+=block
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    if len(line)>2*1024*1024:raise ValueError('Native batch request frame too large')
                    incoming.put(json.loads(line))
                if len(pending)>2*1024*1024:raise ValueError('Native batch request frame too large')
        except Exception as error:input_failure.append(str(error))
        finally:shutdown.set()
    try:
        decoder=create_decoder(folder+'/model.safetensors.index.json',config,capacity,batch,chunk)
        decoder.prefill_compile_ms=0
        if prefix:
            if 'native_hybrid_token_prefix' not in garnet.backend_info():raise ValueError('Native hybrid token prefix extension required')
            decoder.session.configure_token_prefix(profile['prefix_identity'],profile.get('prefix_entries',2),profile.get('prefix_host_bytes',384*1024*1024))
        graphs=decoder._resident_prefill_graphs(chunk)
        if retire_prefill_inputs and any(graph.retired_prefill_hidden_input_slots!=1 for graph in graphs):
            raise ValueError('Frozen prefill input retirement did not cover every layer')
        if 'memory_inventory_path' in profile:
            path=profile['memory_inventory_path']
            if not isinstance(path,str) or not path.startswith('/'):raise ValueError('Inventory output requires an absolute research path')
            descriptors=[dict(tag='decode-'+str(i),object=layer['graph']) for i,layer in enumerate(decoder.layers)]
            descriptors+=[dict(tag='prefill-'+str(i),object=graph) for i,graph in enumerate(graphs)]
            descriptors.append(dict(tag='head',object=decoder.head))
            with open(path,'x') as stream:stream.write(garnet.memory_inventory_json(descriptors,decoder.session))
        policy=garnet.continuous_scheduler();policy.initialize(decoder.session,16)
        if chunked and not policy.set_prefill_chunk(chunk):raise ValueError('Native prefill chunk configuration rejected')
        tokenizer=garnet.tokenizer();tokenizer.open(folder)
        stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
        reader=threading.Thread(target=read,daemon=True);reader.start()
        emit(dict(event='ready',backend=garnet.backend_info(),profile=profile,active_requests=batch,capacity=capacity,
            prefill_chunk=chunk,prefix_cache=prefix,multimodal=False,memory=json.loads(garnet.memory_stats_json())))
        def admit(request):
            key=request.get('id') if isinstance(request,dict) else None
            if not isinstance(key,str) or not key or len(key)>128:raise ValueError('Native batch request identity invalid')
            if key in keys:raise ValueError('Duplicate native batch request identity')
            try:
                body=request['body'];limit=body['max_tokens'];prompt=body['rendered_prompt']
                if not isinstance(prompt,str) or type(limit) is not int or not 2<=limit<=1024:
                    raise ValueError('Native batch request contract invalid')
                ids=list(tokenizer.encode(prompt))
                if not ids or len(ids)+limit>capacity:raise ValueError('Request exceeds the frozen native context profile')
                identity=policy.submit(len(ids),limit,0)
                if identity<=0:raise ValueError('Native scheduler queue or context profile full')
                jobs[identity]=dict(id=key,prompt=ids,limit=limit,generated=[],text_ids=[],cached_tokens=0,began=time.monotonic())
                keys.add(key)
            except Exception as error:emit(dict(event='error',id=key,error=str(error),fatal=False))
        def record_kernel_profile(stage,physical,valid):
            if not kernel_profiles:return
            count=decoder.session.affine_profile_dispatches;misses=decoder.session.affine_profile_misses
            pipelines=decoder.session.affine_profile_pipeline_count;max_threads=decoder.session.affine_profile_max_threads
            expected=profile['expected_head_affine_projections'] if stage=='head' else profile['expected_body_affine_projections']
            if stage=='decode':expected+=profile['expected_head_affine_projections']
            emit(dict(event='native_affine_profile',stage=stage,affine_dispatches=count,misses=misses,
                pipeline_count=pipelines,index_bits=decoder.session.affine_profile_index_bits,pipeline_max_threads=max_threads,active_rows=valid,physical_rows=physical))
            if count!=expected or misses!=0 or not 0<pipelines<=512 or (thread_limit==128 and max_threads!=128) or decoder.session.affine_profile_index_bits!=index_bits:
                raise ValueError('Bounded kernel profile '+stage+' projection coverage or fallback differs')
        def execute(graphs,hidden,positions,slots,active,prediction,epochs,ids):
            if prefix:result=decoder.session.run_tokens(graphs,hidden,positions,slots,active,prediction,epochs,ids)
            else:result=decoder.session.run_leased(graphs,hidden,positions,slots,active,prediction,epochs)
            if kernel_profiles:
                stage='prefill' if len(ids)>batch else ('decode' if prediction>=0 else 'body')
                record_kernel_profile(stage,len(ids),sum(1 for flag in active.tolist() if flag!=0))
            if profile.get('require_mpp',False) and len(ids)>batch:
                count=decoder.session.mpp_affine_dispatches
                if count<1:raise ValueError('Required MPP prefill route was not exercised')
                emit(dict(event='native_mpp',affine_dispatches=count))
            if indexed and len(ids)>batch:
                count=decoder.session.indexed_affine_dispatches
                valid=sum(1 for flag in active.tolist() if flag!=0)
                if valid<len(ids) and count<1:raise ValueError('Required sparse indexed projection route was not exercised')
                emit(dict(event='native_indexed',affine_dispatches=count,active_tokens=valid,physical_rows=len(ids)))
            if reuse64 and len(ids)>batch:
                valid=sum(1 for flag in active.tolist() if flag!=0)
                logical=valid if indexed else len(ids);count=decoder.session.mpp_reuse64_dispatches
                if (logical>=64 and count<1) or (logical<64 and count!=0):
                    raise ValueError('Cooperative64 route disagrees with native token count')
                emit(dict(event='native_reuse64',affine_dispatches=count,logical_rows=logical,physical_rows=len(ids)))
            if active_decode and len(ids)==batch:
                valid=sum(1 for flag in active.tolist() if flag!=0);count=decoder.session.indexed_decode_affine_dispatches
                stage='decode' if prediction>=0 else 'body'
                coverage=profile['expected_body_indexed_decode_projections']+(profile['expected_head_indexed_decode_projections'] if prediction>=0 else 0)
                expected=coverage if valid<batch else 0
                emit(dict(event='native_indexed_decode',stage=stage,affine_dispatches=count,active_rows=valid,physical_rows=batch))
                if count!=expected:raise ValueError('Active FP32 '+stage+' projection coverage '+str(count)+' differs from expected '+str(expected))
            return result
        def predict_rows(epochs):
            result=decoder.session.predict_rows(epochs)
            record_kernel_profile('head',batch,sum(1 for epoch in epochs if epoch!=0))
            if active_decode:
                valid=sum(1 for epoch in epochs if epoch!=0);count=decoder.session.indexed_decode_affine_dispatches
                expected=profile['expected_head_indexed_decode_projections'] if valid<batch else 0
                emit(dict(event='native_indexed_decode',stage='head',affine_dispatches=count,active_rows=valid,physical_rows=batch))
                if count!=expected:raise ValueError('Active FP32 head projection coverage '+str(count)+' differs from expected '+str(expected))
            return result
        def deliver(request,token,stage):
            identity=request['id'];job=jobs[identity]
            if token<0 or token>=vocab:raise ValueError('Native batch output outside vocabulary')
            job['generated'].append(token)
            if token not in stops:job['text_ids'].append(token)
            text=tokenizer.decode(job['text_ids']).encode('utf8').decode('utf8','replace')
            emit(dict(event='token',id=job['id'],token_id=token,sampled_tokens=len(job['generated']),
                prompt_tokens=len(job['prompt']),cached_tokens=job['cached_tokens'],text=text))
            finished=token in stops or len(job['generated'])>=job['limit']
            fn=policy.complete_prefill if stage=='prefill' else policy.complete_decode
            result=fn(identity,token,finished)
            if result!=(3 if finished else 2):raise ValueError('Native scheduler completion disagrees with output budget')
            if finished:
                emit(dict(event='completed',id=job['id'],generated_ids=job['generated'],prompt_ids=job['prompt'],text=text,
                    cached_tokens=job['cached_tokens'],finish_reason='stop' if token in stops else 'length',
                    generation_mode='native_metal_continuous_batch',profile_id=profile['profile_id'],
                    native_request=identity,native_row=request['row'],native_epoch=request['epoch'],
                    request_window_seconds=time.monotonic()-job['began'],memory=json.loads(garnet.memory_stats_json())))
                keys.remove(job['id']);del jobs[identity]
        while not shutdown.is_set():
            if not jobs:
                try:admit(incoming.get(timeout=.1))
                except queue.Empty:continue
                # Bounded coalescing is part of the frozen serving profile and
                # included in client TTFT, not a benchmark warmup adjustment.
                deadline=time.monotonic()+.005
                while time.monotonic()<deadline and not shutdown.is_set():
                    try:admit(incoming.get(timeout=max(.0001,deadline-time.monotonic())))
                    except queue.Empty:break
            while not incoming.empty():admit(incoming.get_nowait())
            if not jobs:continue
            tick=json.loads(policy.next_json());plans=tick['batches']
            if not plans:raise ValueError('Native scheduler has live jobs but made no progress')
            if mixed and len(plans)==2 and {plan['stage'] for plan in plans}=={'decode','prefill'}:
                # One existing masked chunk graph executes all physical rows.
                # Native policy supplies leases and progress; this bridge only
                # materializes inputs. Projection rows still include padding.
                packed=tick.get('packed')
                if packed is None or packed.get('execution')!='metadata_only':
                    raise ValueError('Mixed forward requires the native packed segment contract')
                rows={};epochs=[0]*batch;final_epochs=[0]*batch
                for plan in plans:
                    emit(dict(event='native_batch',stage=plan['stage'],active_rows=plan['active_rows'],requests=plan['requests']))
                    for request in plan['requests']:
                        row=request['row']
                        if row in rows:raise ValueError('Mixed native tick repeats a physical row')
                        rows[row]=(plan['stage'],request);epochs[row]=request['epoch']
                        if plan['stage']=='decode' or request['prefill_offset']+request['prefill_tokens']==len(jobs[request['id']]['prompt']):
                            final_epochs[row]=request['epoch']
                ids=[];flags=[];slots=[]
                for row in range(batch):
                    entry=rows.get(row)
                    for t in range(chunk):
                        stage,request=entry if entry else (None,None)
                        count=(1 if stage=='decode' else request['prefill_tokens']) if entry else 0
                        valid=t<count
                        position=request['slot']+t if valid else 0
                        token=(request['token_id'] if stage=='decode' else jobs[request['id']]['prompt'][position]) if valid else 0
                        ids.append(token);flags.append(float(valid));slots.append(float(position))
                actual_tokens=sum(int(flag) for flag in flags)
                if actual_tokens!=packed['token_count']:raise ValueError('Mixed input count disagrees with native packed plan')
                hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(batch)]) for t in range(chunk)])
                execute(graphs,hidden,f32([slot for slot in slots for axis in range(3)],[batch,chunk,3]),
                    f32(slots,[batch,chunk]),f32(flags,[batch,chunk]),-1,epochs,ids)
                predicted=predict_rows(final_epochs)
                emit(dict(event='native_mixed_forward',segments=packed['segments'],actual_tokens=actual_tokens,
                    projection_rows=batch*chunk,scope='One masked chunk forward; projections retain padding'))
                for row,(stage,request) in rows.items():
                    if final_epochs[row]:deliver(request,predicted[row],stage)
                    elif not policy.complete_prefill_chunk(request['id'],request['prefill_tokens']):
                        raise ValueError('Native mixed prefill completion rejected')
                    if stage=='prefill':
                        emit(dict(event='native_prefill_chunk',native_request=request['id'],row=row,epoch=request['epoch'],
                            offset=request['prefill_offset'],tokens=request['prefill_tokens'],final=bool(final_epochs[row])))
                continue
            for plan in plans:
                if shutdown.is_set():break
                emit(dict(event='native_batch',stage=plan['stage'],active_rows=plan['active_rows'],requests=plan['requests']))
                if plan['stage']=='prefill' and chunked:
                    requests={request['row']:request for request in plan['requests']}
                    ids=[];flags=[];slots=[];final_epochs=[0]*batch
                    for row in range(batch):
                        request=requests.get(row)
                        if request is not None:
                            offset=request['prefill_offset'];count=request['prefill_tokens'];job=jobs[request['id']]
                            if not 0<count<=chunk or offset<0 or offset+count>len(job['prompt']):raise ValueError('Native prefill chunk plan out of bounds')
                            if offset+count==len(job['prompt']):final_epochs[row]=request['epoch']
                        for t in range(chunk):
                            valid=request is not None and t<request['prefill_tokens']
                            position=request['prefill_offset']+t if valid else 0
                            ids.append(jobs[request['id']]['prompt'][position] if valid else 0)
                            flags.append(1.0 if valid else 0.0);slots.append(float(position))
                    hidden=decoder.io.stack_tokens([decoder.io.embedding([ids[row*chunk+t] for row in range(batch)]) for t in range(chunk)])
                    execute(graphs,hidden,f32([slot for slot in slots for axis in range(3)],[batch,chunk,3]),
                        f32(slots,[batch,chunk]),f32(flags,[batch,chunk]),-1,plan['epochs'],ids)
                    predicted=predict_rows(final_epochs) if any(final_epochs) else None
                    for request in plan['requests']:
                        final=final_epochs[request['row']]!=0
                        if final:deliver(request,predicted[request['row']],'prefill')
                        elif not policy.complete_prefill_chunk(request['id'],request['prefill_tokens']):raise ValueError('Native prefill chunk completion rejected')
                        emit(dict(event='native_prefill_chunk',native_request=request['id'],row=request['row'],epoch=request['epoch'],
                            offset=request['prefill_offset'],tokens=request['prefill_tokens'],final=final))
                    continue
                if plan['stage']=='prefill':
                    rows={request['row']:jobs[request['id']] for request in plan['requests']}
                    for request in plan['requests']:
                        job=jobs[request['id']]
                        if prefix:job['cached_tokens']=decoder.session.restore_token_prefix(request['row'],request['epoch'],job['prompt'])
                    remaining=max(len(job['prompt'])-job['cached_tokens'] for job in rows.values())
                    for offset in range(0,remaining,chunk):
                        if shutdown.is_set():break
                        ids=[];flags=[];slots=[];by_token=[]
                        for t in range(chunk):
                            by_token.append([rows[row]['prompt'][rows[row]['cached_tokens']+offset+t]
                                if row in rows and rows[row]['cached_tokens']+offset+t<len(rows[row]['prompt']) else 0 for row in range(batch)])
                        for row in range(batch):
                            job=rows.get(row)
                            for t in range(chunk):
                                position=job['cached_tokens']+offset+t if job else 0
                                valid=job is not None and position<len(job['prompt'])
                                ids.append(by_token[t][row]);flags.append(1.0 if valid else 0.0);slots.append(float(position) if valid else 0.0)
                        positions=[slot for slot in slots for axis in range(3)]
                        execute(graphs,decoder.io.stack_tokens([decoder.io.embedding(tokens) for tokens in by_token]),
                            f32(positions,[batch,chunk,3]),f32(slots,[batch,chunk]),f32(flags,[batch,chunk]),-1,plan['epochs'],ids)
                    if shutdown.is_set():break
                    predicted=predict_rows(plan['epochs'])
                    if prefix:
                        for request in plan['requests']:
                            job=jobs[request['id']]
                            if job['cached_tokens']<len(job['prompt']):decoder.session.cache_token_prefix(request['row'],request['epoch'])
                else:
                    ids=[0]*batch;slots=[0.0]*batch
                    for request in plan['requests']:ids[request['row']]=request['token_id'];slots[request['row']]=float(request['slot'])
                    positions=[slot for slot in slots for axis in range(3)]
                    predicted=execute([layer['graph'] for layer in decoder.layers],decoder.io.embedding(ids),f32(positions,[batch,3]),
                        f32(slots,[batch]),f32([float(value) for value in plan['active_rows']],[batch]),0,plan['epochs'],ids)
                for request in plan['requests']:deliver(request,predicted[request['row']],plan['stage'])
        if input_failure:raise ValueError(input_failure[0])
    except Exception as error:
        for job in jobs.values():emit(dict(event='error',id=job['id'],error=str(error),fatal=True))
        raise
    finally:
        shutdown.set()
        try:
            if policy is not None:policy.close()
        finally:
            try:
                if decoder is not None:decoder.close()
                emit(dict(event='closed',memory=json.loads(garnet.memory_stats_json())))
            finally:garnet.shutdown()

main()
