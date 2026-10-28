"""Ordinary original-model early decode state snapshots; no observer graph or timing."""
from pathlib import Path
import base64,hashlib,json,os,sys,zipfile
sha=lambda b:hashlib.sha256(b).hexdigest()

def main():
    root,out,mode,reference=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]);assert mode in ('reference','candidate') and not out.exists()
    m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/b4-full4k-decode-prefix-preparation.json').read_bytes());batch=1 if mode=='reference' else 4
    expected=p[mode+'_installed_files'];library=sha((root/'build/bin/garnet.dll').read_bytes());assert library==p[mode+'_library_sha256']
    for n,h in expected.items():assert sha((root/n).read_bytes())==h,('Input drift',n)
    for n,h in p['runtime'].items():assert sha((root/'build/bin'/n).read_bytes())==(library if n=='garnet.dll' else h)
    for n,h in p['checkpoints'].items():assert sha((root/n).read_bytes())==h
    assert {k:os.environ.get(k) for k in p['flags']}=={**p['flags'],'GARNET_FP8_BATCH4_PREFILL_CANONICAL':'0' if mode=='reference' else '1','GARNET_FP8_GDN_BATCH4_DECODE_CANONICAL':'0' if mode=='reference' else '1','GARNET_FP8_V_BATCH4_DECODE_CANONICAL':'0' if mode=='reference' else '1','GARNET_FP8_QK_BATCH4_DECODE_CANONICAL':'0' if mode=='reference' else '1'}
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')]
    import garnet
    from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
    from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder,create_cuda_vision,prepare_cuda_image
    from test2026.cuda.qwen38_5090.native_prompt import image_positions
    from test2026.cuda.qwen38_5090.paged_prompt_rows import PagedPromptRows
    from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
    from test2026.cuda.qwen38_5090.b4_b1_exact_gate import compare_capture,require_ids
    from test2026.cuda.qwen38_5090.b4_full4k_inputs import load_corpus
    corpus=(m/'evidence/fixed-length-4096-corpus.zip').read_bytes();assert sha(corpus)==p['input_corpus_sha256'];selection=load_corpus(corpus);keys=list(selection['inputs']);assert len(keys)==8
    bodies=[selection['inputs'][key]['body'] for key in keys];inputs=[]
    for key,body in zip(keys,bodies):
        row=dict(selection['inputs'][key]);content=body['messages'][0]['content'];row['image_sha256']=None if isinstance(content,str) else sha(base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True));inputs.append(row)
    assert all(len(row['prompt_ids'])==3968 for row in inputs)
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';cfg=json.loads((checkpoint/'config.json').read_bytes());out.mkdir();decoder=vision=None
    report=dict(status='constructing',mode=mode,physical_batch=batch,capacity=4096,prefill_chunk=32,decode_chunk=4,library_sha256=library,inputs=selection['inputs'],rows=[],performance_admitted=False,full4k_MTP_quality_qualified=False,MTP_quality_qualified=False)
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        if batch==1:decoder=create_cuda_bf16_low_batch_study_decoder(str(checkpoint),cfg)
        else:
            decoder=CUDAPackedDecoder(str(checkpoint),cfg['text_config'],4,4096,32,4,flat_concat=True,external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,host_embedding_lookup=True,joined_mlp_projections=True,shared_output_storage=True,register_delta_128=True,bf16_kv_state=True)
            decoder.io.enable_nvfp4_head(True)
        decoder.enable_cuda_graph(True);io=decoder.io;report['graph_stats']={str(t):json.loads(v.graph.stats_json()) for t,v in decoder.programs.items()};report['pool']=json.loads(io.memory_json());save()
        assert sorted(decoder.programs)==[4,32] and report['pool']['embedding_mode']=='host_mapped_bf16' and report['pool']['head_precision']=='modelopt_nvfp4_w4a4'
        assert (report['pool']['compressed_projection_pool_entries'],report['pool']['compressed_nvfp4_entries'],report['pool']['compressed_fp8_entries'])==(336,128,208)
        counts={'native_fp8_m128_canonical_projections':48,'native_fp8_z128_canonical_projections':0,'native_fp8_z128_reconstructed_projections':48,'native_fp8_kv128_canonical_projections':32,'native_fp8_q128_canonical_projections':16,'native_fp8_o128_canonical_projections':16,'native_full_chunk_stage_witnesses':0}
        for t,stats in report['graph_stats'].items():
            for key,count in counts.items():assert type(stats[key]) is int and stats[key]==(count if batch==4 and t=='32' else 0),(t,key,stats.get(key),count)
        vision=create_cuda_vision(str(checkpoint),cfg,16,16);vision.graph.enable_cuda_graph(True);report['vision_stats']=json.loads(vision.graph.stats_json());assert all(report['vision_stats'][key]==0 for key in counts)
        tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';report['status']='full4k_cases_running';save()
        def snapshot(label):
            assert decoder.state_transaction is None and not decoder.poisoned
            logical=logical_state_witness(io,decoder.state,decoder.programs[4].layout,batch,106823680,decoder.slots,packed_bf16=True);vectors={}
            for kind,tensor in (('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))):
                raw=bytes(io.finite_f32_bytes(tensor));assert len(raw)==batch*20480;name=label+'-'+kind+'.f32';(out/name).write_bytes(raw);vectors[kind]=dict(file=name,bytes=len(raw),sha256=sha(raw))
            return dict(logical=logical,vectors=vectors)
        index={key:i for i,key in enumerate(keys)}
        selected=[index[key] for key in selection['groups'][2]['keys']]
        groups=[[i] for i in selected] if batch==1 else [selected]
        for group_index,physical in enumerate(groups):
            decoder.reset();record=dict(physical_rows=physical,keys=[keys[i] for i in physical],label=selection['groups'][2]['label'] if batch==4 else 'B1-'+keys[physical[0]],reset_state=snapshot('case'+str(group_index)+'-reset'));report['rows'].append(record);save();prepared=[]
            for index in physical:
                row=inputs[index];body=bodies[index];content=body['messages'][0]['content'];visual=None;start=-1
                if isinstance(content,str):ids=tokenizer.encode('<|im_start|>user\n'+content+suffix);positions=[[i]*3 for i in range(len(ids))];current=len(ids);assert row['image_sha256'] is None
                else:
                    data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True);assert sha(data)==row['image_sha256'];image=out/('case'+str(group_index)+'-input'+str(index)+'.jpg');image.write_bytes(data)
                    patches=prepare_cuda_image(io,str(checkpoint),str(image));assert [patches[1],patches[2]]==[16,16];visual=io.clone(vision.encode_patches(patches[0]));leading=tokenizer.encode('<|im_start|>user\n<|vision_start|>');start=len(leading)
                    ids=leading+[cfg['image_token_id']]*64+tokenizer.encode('<|vision_end|>'+content[1]['text']+suffix);positions,current=image_positions(len(ids),start,64,8)
                assert ids==row['prompt_ids'];prepared.append(dict(ids=ids,positions=positions,current=current,visual=visual,start=start))
            lengths=[len(v['ids']) for v in prepared];n=max(lengths);positions=[v['positions']+[[0,0,0]]*(n-len(v['ids'])) for v in prepared];current=[v['current'] for v in prepared]
            hidden=PagedPromptRows(io,[v['ids'] for v in prepared],[v['visual'] for v in prepared],[v['start'] for v in prepared],page_size=128)
            try:prediction=decoder.prefill_sequence(hidden,positions,lengths);record['prompt_cache']=hidden.stats()
            finally:hidden.close()
            record['prefill_state']=snapshot('case'+str(group_index)+'-prefill');record['generated_ids']=[[] for _ in physical];record['post_step_states']=[];save()
            for step in range(8):
                assert len(prediction)==batch
                for row,token in enumerate(prediction):assert type(token) is int and 0<=token<248320;record['generated_ids'][row].append(token)
                if step+1<8:
                    prediction=decoder.step(prediction,[1]*batch,[[v]*3 for v in current]);current=[v+1 for v in current]
                    record['post_step_states'].append(dict(step=step+1,capture=snapshot('case'+str(group_index)+'-step'+str(step+1))));save()
            record['final_state']=record['post_step_states'][-1]['capture'];assert decoder.slots==[v+7 for v in lengths];decoder.reset();record['final_reset']=snapshot('case'+str(group_index)+'-final-reset');save();print('B4_FULL4K_DECODE_PREFIX_COLLECTED',mode,group_index,flush=True)
        # Raw capture only: the exclusive CPython owner enforces the unchanged full gates.
        report['status']='full4k_first_seven_decode_steps_captured_unqualified'
        save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if vision is not None:vision.close()
        if decoder is not None:decoder.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
