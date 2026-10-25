"""Ordinary-only full-cache reference corpus; no MTP or timing admission."""
import base64,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True);path=out/'report.json';assert not path.exists()
sha=lambda b:hashlib.sha256(b).hexdigest();checkpoint=root/'models/Qwen3.8-27B-NVFP4';config_raw=(checkpoint/'config.json').read_bytes();config=json.loads(config_raw)
assert os.environ.get('GARNET_NVFP4_DOWN_CANDIDATE')=='0' and os.environ.get('GARNET_FP8_GDN_M32_CANONICAL')=='1' and os.environ.get('GARNET_FP8_FULL_ATTN_M32_CANONICAL')=='1'
keys=['s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green'];cases=[]
for key in keys:
    base=root/'results/native-mtp3-http';req=(base/(key+'-request.json')).read_bytes();response=(base/(key+'-response.json')).read_bytes()
    cases.append({'key':key,'body':json.loads(req),'choice':json.loads(response)['choices'][0],
        'request_sha256':sha(req),'response_sha256':sha(response)})
report={'status':'constructing_reference','batch':4,'capacity':4096,'prefill_chunk':32,'decode_chunk':4,'bf16_kv_float_reference':True,
    'library_sha256':sha((root/'build/bin/garnet.dll').read_bytes()),'model_config_sha256':sha(config_raw),'cases':[],
    'oracles':[{n:c[n] for n in ('key','request_sha256','response_sha256')} for c in cases],
    'scope':'Ordinary-only independent dense FP32-cache/BF16-value reference on pinned mixed checkpoint. Four fixed text/image keys, two groups repeated twice, each original output limit and stops. Complete logical state hashes and raw final hidden/normalized vectors. Old FP32 IDs retained as failed/different observations, not redefined as passed. This is a prospective same-precision arithmetic oracle; no broad quality, packed/MTP/serving or speed admission.'}
def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
save();decoder=None;vision=None
try:
    decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],4,4096,32,4,flat_concat=True,external_compressed_weights=True,
        retain_linear_inputs=True,deferred_state_commit=True,host_embedding_lookup=True,joined_mlp_projections=True,
        shared_output_storage=True,register_delta_128=True,bf16_kv_float_reference=True)
    assert decoder.delta_commit is None and decoder.bf16_prefill_commit is None and not decoder.bf16_kv_state and not decoder.kv_delta_state
    assert all(p.bf16_kv_float_reference and p.delta_layout is None and p.state_width==173932544 for p in decoder.programs.values())
    decoder.enable_cuda_graph(True);io=decoder.io;pool=json.loads(io.memory_json())
    assert pool['head_precision']=='modelopt_nvfp4_w4a4' and pool['embedding_mode']=='host_mapped_bf16'
    assert (pool['compressed_projection_pool_entries'],pool['compressed_nvfp4_entries'],pool['compressed_fp8_entries'])==(336,128,208)
    report['pool']=pool;save();vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
    tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
    program=decoder.programs[4]
    def state():
        assert decoder.state_transaction is None and not decoder.poisoned
        return logical_state_witness(io,decoder.state,program.layout,4,program.state_width,decoder.slots,packed_bf16=False)
    for group_index,group in enumerate([cases[:2]*2,cases[2:]*2]):
        for repeat in range(2):
            decoder.reset();rows=[]
            for row,c in enumerate(group):
                ids=c['choice']['prompt_token_ids'];content=c['body']['messages'][0]['content'];visual=None;start=-1
                if isinstance(content,list):
                    assert content[0]['type']=='image_url' and content[1]['type']=='text'
                    data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True)
                    image=root/'tmp'/('bf16-reference-full-row-'+str(row)+'.jpg');image.write_bytes(data)
                    prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16]
                    visual=io.clone(vision.encode_patches(prepared[0]));start=ids.index(config['image_token_id']);assert ids.count(config['image_token_id'])==64
                    positions,current=image_positions(len(ids),start,64,8)
                else:positions=[[i]*3 for i in range(len(ids))];current=len(ids)
                embeddings=io.embedding(ids)
                hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
                rows.append({'ids':ids,'positions':positions,'current':current,'hidden':hidden,'visual':visual,'embeddings':embeddings})
            lengths=[len(r['ids']) for r in rows];count=max(lengths);zero=io.zeros([1,5120])
            hidden=[io.stack_rows([r['hidden'][t] if t<len(r['ids']) else zero for r in rows]) for t in range(count)]
            positions=[r['positions']+[[0,0,0]]*(count-len(r['ids'])) for r in rows];limits=[c['body']['max_tokens'] for c in group]
            assert len(set(limits))==1 and all(type(n) is int and 1<=n<=128 for n in limits)
            prediction=decoder.prefill_sequence(hidden,positions,lengths);prefill=state();generated=[[] for _ in rows];active=[1]*4;current=[r['current'] for r in rows]
            for step in range(limits[0]):
                for row in range(4):
                    if active[row]:
                        generated[row].append(prediction[row])
                        if prediction[row] in stops or len(generated[row])==limits[row]:active[row]=0
                if not any(active):break
                prediction=decoder.step(prediction,active,[[v]*3 for v in current]);current=[v+flag for v,flag in zip(current,active)]
            final=state();raw={}
            for label,value in [('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))]:
                data=bytes(io.finite_f32_bytes(value));assert len(data)==81920 and data[:40960]==data[40960:]
                filename='group-'+str(group_index)+'-repeat-'+str(repeat)+'-reference-'+label+'.f32';(out/filename).write_bytes(data)
                raw[label]={'file':filename,'bytes':len(data),'sha256':sha(data)}
            assert generated[:2]==generated[2:] and all(s['logical_row_sha256'][:2]==s['logical_row_sha256'][2:] for s in (prefill,final))
            record={'group':group_index,'repeat':repeat,'keys':[c['key'] for c in group],'prompt_lengths':lengths,'limits':limits,
                'generated_ids':generated,'retained_fp32_ids':[c['choice']['response_token_ids'] for c in group],
                'retained_fp32_ids_exact':generated==[c['choice']['response_token_ids'] for c in group],
                'prefill_state':prefill,'final_state':final,'raw':raw,'memory':decoder.memory_info()}
            report['cases'].append(record);save()
            if repeat:
                previous=report['cases'][-2]
                assert previous['generated_ids']==record['generated_ids'] and previous['prefill_state']==prefill and previous['final_state']==final
                assert all(previous['raw'][label]['sha256']==raw[label]['sha256'] for label in raw)
            print('BF16_FULL_REFERENCE_CASE_FINISHED',group_index,repeat,flush=True)
    report['status']='bf16_full_reference_corpus_finished';save()
except BaseException as error:report.update(status='failed',error=repr(error));save();raise
finally:
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
