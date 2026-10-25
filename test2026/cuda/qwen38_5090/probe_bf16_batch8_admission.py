"""Actual B8 model/vision/full4K admission, unrun; no performance measurements.

An outer owner must first observe the original HTTP supervisor terminal, check
idle GPU/source/runtime identity, monitor memory and preserve the raw attempt.
No MTP, HTTP serving, broad quality or speed qualification is performed here.
"""
from pathlib import Path
import base64,hashlib,json,os,sys,zipfile
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
from test2026.cuda.qwen38_5090.bf16_batch8_admission import ORDERS,compare_permuted,compare_vector_rows

def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists();out.mkdir()
    path=out/'report.json';sha=lambda raw:hashlib.sha256(raw).hexdigest();checkpoint=root/'models/Qwen3.8-27B-NVFP4'
    config=json.loads((checkpoint/'config.json').read_bytes());library=sha((root/'build/bin/garnet.dll').read_bytes())
    model='tokenizer-nfc-model-consistency-attempt1';proof=json.loads((root/'results'/(model+'-verified.json')).read_bytes())
    assert proof['status']=='tokenizer_nfc_model_consistency_replayed' and proof['library_sha256']==library
    assert sha((root/'results'/(model+'-evidence.zip')).read_bytes())==proof['archive_sha256']
    for flag in ('GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M32_CANONICAL','GARNET_PARALLEL_BATCH_TOP1'):assert os.environ.get(flag)=='1'
    report={'status':'constructing','physical_batch':8,'capacity':4096,'library_sha256':library,'cases':[],
            'mtp_executed':False,'http_executed':False,'quality_qualified':False,'performance_admitted':False,
            'scope':'Existing opt-in B8 BF16-cache/T32prefill/T4decode mixedNVFP4/FP8 model. Ordinary fixed-count8short/128full4K generated tokens, original/rotated/reset-repeat row order. Complete state-component hashes and raw hidden/normalized vectors compare after inverse permutation. Construction/vision/prompt stacking/full4K memory must actually fit. No MTP, broad805quality, HTTP batching or performance admission.'}
    def save():path.write_text(json.dumps(report,indent=2),encoding='utf8')
    decoder=vision=None;save()
    try:
        decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],8,4096,32,4,flat_concat=True,
            external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,
            host_embedding_lookup=True,joined_mlp_projections=True,shared_output_storage=True,
            bf16_kv_state=True,register_delta_128=True)
        decoder.io.enable_nvfp4_head(True);decoder.enable_cuda_graph(True);io=decoder.io
        pool=json.loads(io.memory_json());report['pool_before']=pool;report['memory_after_decoder_construction']=decoder.memory_info();save()
        assert pool['embedding_mode']=='host_mapped_bf16' and pool['embedding_table_bytes']==0 and pool['host_embedding_mapped_tensor_bytes']==2542796800
        assert pool['compressed_projection_pool_entries']==336 and pool['compressed_nvfp4_entries']==128 and pool['compressed_fp8_entries']==208
        assert pool['head_precision']=='modelopt_nvfp4_w4a4' and pool['batch_top1_parallel_enabled']
        assert decoder.batch==8 and decoder.programs[4].state_width==106823680 and decoder.bf16_kv_state and not decoder.bf16_kv_low_batch_state
        report['commit_helper']=json.loads(decoder.delta_commit.stats_json());report['prefill_commit_helper']=json.loads(decoder.bf16_prefill_commit.stats_json());save()
        assert report['commit_helper']['ready'] and report['prefill_commit_helper']['ready']
        vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
        tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
        suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        def snapshot(label):
            assert decoder.state_transaction is None and not decoder.poisoned
            witness=logical_state_witness(io,decoder.state,decoder.programs[4].layout,8,106823680,decoder.slots,packed_bf16=True)
            vectors={};raws={}
            for kind,value in (('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))):
                raw=bytes(io.finite_f32_bytes(value));assert len(raw)==8*5120*4;name=label+'-'+kind+'.f32';(out/name).write_bytes(raw)
                vectors[kind]={'file':name,'bytes':len(raw),'sha256':sha(raw)};raws[kind]=raw
            return {'logical':witness,'vectors':vectors},raws
        decoder.reset();empty,_=snapshot('empty');report['empty_state']=empty;save()
        oracle=root/'results/native-mtp3-http';short=[]
        for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green'):
            wire=(oracle/(key+'-request.json')).read_bytes();response=json.loads((oracle/(key+'-response.json')).read_bytes())
            short.append((key,json.loads(wire),response['choices'][0]['prompt_token_ids'],sha(wire)))
        groups=[('short-mixed',short+short,8)]
        corpus=root/'Garnet/test2026/cuda/qwen38_5090/evidence/fixed-length-4096-corpus.zip'
        assert sha(corpus.read_bytes())=='596991dd0907ccb8ae8ffb090737f59ba8da3faac181bb18fadc74ae0e4ef3ce'
        with zipfile.ZipFile(corpus) as z:
            manifest=json.loads(z.read('manifest.json'));rows={row['key']:row for row in manifest['rows']}
            for case in ('text','image'):
                cohort=next(c for c in manifest['cohorts'] if c['case']==case and c['concurrency']==8);prepared=[]
                for key in cohort['measured']:
                    row=rows[key];wire=z.read(row['request_file']);ids=z.read(row['expected_prompt_ids_file'])
                    assert sha(wire)==row['request_sha256'] and sha(ids)==row['expected_prompt_ids_sha256'];body=json.loads(wire)
                    assert body['ignore_eos'] is True and len(json.loads(ids))==3968
                    prepared.append((key,body,json.loads(ids),sha(wire)))
                assert len(prepared)==8;groups.append(('full4k-'+case,prepared,128))
        for label,inputs,limit in groups:
            prepared=[]
            for row,(key,body,expected,digest) in enumerate(inputs):
                content=body['messages'][0]['content'];visual=None;start=-1;image_hash=None
                if isinstance(content,str):
                    ids=tokenizer.encode('<|im_start|>user\n'+content+suffix);positions=[[i]*3 for i in range(len(ids))];current=len(ids)
                else:
                    data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True);image_hash=sha(data);image=out/(label+'-row'+str(row)+'.jpg');image.write_bytes(data)
                    patches=prepare_cuda_image(io,str(checkpoint),str(image));assert [patches[1],patches[2]]==[16,16]
                    visual=io.clone(vision.encode_patches(patches[0]));leading=tokenizer.encode('<|im_start|>user\n<|vision_start|>');start=len(leading)
                    ids=leading+[config['image_token_id']]*64+tokenizer.encode('<|vision_end|>'+content[1]['text']+suffix)
                    positions,current=image_positions(len(ids),start,64,8)
                assert ids==expected and len(ids)+limit<=4096
                embeddings=io.embedding(ids);hidden=[io.visual_token(visual,i-start) if visual is not None and start<=i<start+64 else io.row(embeddings,i) for i in range(len(ids))]
                prepared.append({'key':key,'request_sha256':digest,'prompt_ids':ids,'positions':positions,'current':current,'image_sha256':image_hash,'embeddings':embeddings,'hidden':hidden,'visual':visual})
            reference=None;reference_raw=None
            for repeat,order in enumerate(ORDERS):
                stem=label+'-order'+str(repeat);decoder.reset();reset,_=snapshot(stem+'-reset');assert compare_permuted(empty['logical'],reset['logical'],tuple(range(8)))
                physical=[prepared[i] for i in order];count=max(len(row['prompt_ids']) for row in physical);zero=io.zeros([1,5120])
                hidden=[io.stack_rows([row['hidden'][t] if t<len(row['prompt_ids']) else zero for row in physical]) for t in range(count)]
                positions=[row['positions']+[[0,0,0]]*(count-len(row['prompt_ids'])) for row in physical]
                lengths=[len(row['prompt_ids']) for row in physical];current=[row['current'] for row in physical]
                prediction=decoder.prefill_sequence(hidden,positions,lengths);prefill,prefill_raw=snapshot(stem+'-prefill')
                generated=[[] for _ in range(8)]
                for step in range(limit):
                    assert len(prediction)==8
                    for row,token in enumerate(prediction):
                        assert type(token) is int and 0<=token<248320;generated[row].append(token)
                    if step+1<limit:prediction=decoder.step(prediction,[1]*8,[[position]*3 for position in current]);current=[n+1 for n in current]
                final,final_raw=snapshot(stem+'-final');assert decoder.slots==[n+limit-1 for n in lengths]
                record={'group':label,'repeat':repeat,'order':list(order),'actual_limit':limit,'inputs':[{'key':row['key'],'request_sha256':row['request_sha256'],'prompt_ids':row['prompt_ids'],'image_sha256':row['image_sha256']} for row in physical],
                        'generated_ids':generated,'text':[tokenizer.decode([n for n in row if n not in stops]) for row in generated],
                        'reset_state':reset,'prefill_state':prefill,'final_state':final,'memory_after_case':decoder.memory_info(),
                        'inverse_permutation_exact':False,'performance_admitted':False}
                report['cases'].append(record);save()
                if reference is None:reference=record;reference_raw=(prefill_raw,final_raw)
                assert all(generated[row]==reference['generated_ids'][original] for row,original in enumerate(order))
                assert compare_permuted(reference['prefill_state']['logical'],prefill['logical'],order) and compare_permuted(reference['final_state']['logical'],final['logical'],order)
                for kind in ('hidden','normalized'):
                    assert compare_vector_rows(reference_raw[0][kind],prefill_raw[kind],order) and compare_vector_rows(reference_raw[1][kind],final_raw[kind],order)
                record['inverse_permutation_exact']=True;save();print('B8_ADMISSION_CASE_PASS',label,repeat,flush=True)
                del hidden
            del prepared,physical,embeddings,visual,zero
        decoder.reset();final_reset,_=snapshot('final-reset');assert compare_permuted(empty['logical'],final_reset['logical'],tuple(range(8)))
        report['final_reset']=final_reset;report['memory_after']=decoder.memory_info();report['pool_after']=json.loads(io.memory_json());report['status']='bf16_batch8_ordinary_admission_collected_needs_replay';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if vision is not None:vision.close()
        if decoder is not None:decoder.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
