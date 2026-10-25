"""Explicit real full4K correctness for the short-model-qualified MTP T16 study."""
from pathlib import Path
import base64,hashlib,json,os,sys,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    root,out=map(Path,sys.argv[1:3]);assert not out.exists();m=root/'Garnet/test2026/cuda/qwen38_5090';repo=root/'Garnet';name='mtp-prefill16-model-consistency-attempt1'
    proof=json.loads((root/'results'/(name+'-verified.json')).read_bytes());archive=root/'results'/(name+'-evidence.zip');assert proof['status']=='mtp_prefill16_short_target_model_replayed' and sha(archive.read_bytes())==proof['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        closed=json.loads(z.read('results/'+name+'/workflow.json'));assert closed==json.loads((root/'results'/name/'workflow.json').read_bytes()) and closed['exit_code']==0 and closed['cleanup_finished'] and closed['library_unchanged']
        assert closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']
        for path,digest in closed['source_sha256'].items():assert sha((root/path).read_bytes())==digest,('Qualified source drift',path)
        for path,digest in closed['runtime_dependencies'].items():assert sha((root/'build/bin'/path).read_bytes())==digest
    for flag in ('GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M32_CANONICAL','GARNET_PARALLEL_BATCH_TOP1'):assert os.environ.get(flag)=='1'
    cases_path=m/'evidence/native-host-stage-cases.json';p=json.loads(cases_path.read_bytes());parent='fixed-length-http-study-attempt1';parent_proof=root/'results'/(parent+'-verified.json');q=json.loads(parent_proof.read_bytes())
    assert p['status']=='native_host_stage_retained_cases_prepared_gpu_unrun' and q['status']=='fixed_length_http_study_replayed' and q['native_fixed_4k_metrics_replayed'] and sha(parent_proof.read_bytes())==p['parent_proof_sha256'] and sha((root/'results'/(parent+'-evidence.zip')).read_bytes())==q['archive_sha256']==p['parent_archive_sha256']
    library=sha((root/'build/bin/garnet.dll').read_bytes());assert library==p['library_sha256']==proof['library_sha256'];cases=p['cases'];assert len(cases)==4
    sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(repo))
    import garnet
    from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder,create_cuda_vision,prepare_cuda_image
    from xModel.qwen35.vl_0_8b.cuda_mtp_prefill16_study import CUDAMTPPrefill16StudySession
    from test2026.cuda.qwen38_5090.native_prompt import image_positions
    from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
    from test2026.cuda.qwen38_5090.mtp_prefill16_full4k import ORDER,replay_report
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes());out.mkdir();decoder=vision=session=None
    report={'status':'constructing','physical_batch':1,'capacity':4096,'library_sha256':library,'short_model_archive_sha256':proof['archive_sha256'],'cases_sha256':sha(cases_path.read_bytes()),'cases':[],'quality_qualified':False,'performance_admitted':False,'scope':'Explicit wider MTP prefill actual4K target-state/ID/reset/callback/memory qualification only. Six cases include four retained text/image3968+128 inputs and first-text/image repeats. Ordinary target source/runtime unchanged, T4 decode/verifier unchanged. No broad805quality, HTTP or speed admission.'}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    save()
    try:
        decoder=create_cuda_bf16_low_batch_study_decoder(str(checkpoint),config);decoder.enable_cuda_graph(True);io=decoder.io;vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
        session=CUDAMTPPrefill16StudySession(decoder,str(checkpoint),[],3,head_candidates=0,prefix_replay=True,draft_chunk_size=4,draft_builder_workspace_bytes=128<<20)
        tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));width=decoder.programs[4].state_width
        report['memory_after_construction']=session.memory_info();report['pool_before']=json.loads(io.memory_json());report['status']='cases_running';save()
        def state(label):
            assert decoder.state_transaction is None and not session.verifier.pending and not decoder.poisoned
            logical=logical_state_witness(io,decoder.state,decoder.programs[4].layout,1,width,decoder.slots,packed_bf16=True);vectors={}
            for kind,value in (('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))):
                raw=bytes(io.finite_f32_bytes(value));assert len(raw)==20480;file=label+'-'+kind+'.f32';(out/file).write_bytes(raw);vectors[kind]={'file':file,'bytes':len(raw),'sha256':sha(raw)}
            return {'logical':logical,'vectors':vectors}
        def same(a,b):return a['logical']==b['logical'] and all(a['vectors'][n]['sha256']==b['vectors'][n]['sha256'] for n in ('hidden','normalized'))
        session.reset();report['empty_state']=state('empty');save();suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        for sequence,index in enumerate(ORDER):
            case=cases[index];body=case['body'];assert body['ignore_eos'] is True and body['max_tokens']==128 and body['temperature']==0 and body['chat_template_kwargs']=={'enable_thinking':False};content=body['messages'][0]['content'];visual=None;start=-1
            if isinstance(content,str):ids=tokenizer.encode('<|im_start|>user\n'+content+suffix);positions=[[i]*3 for i in range(len(ids))];current=len(ids)
            else:
                data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True);assert sha(data)==case['image_sha256'];image=out/('case-'+str(index)+'.jpg')
                if image.exists():assert image.read_bytes()==data
                else:image.write_bytes(data)
                prepared=prepare_cuda_image(io,str(checkpoint),str(image));assert [prepared[1],prepared[2]]==[16,16];visual=io.clone(vision.encode_patches(prepared[0]));leading=tokenizer.encode('<|im_start|>user\n<|vision_start|>');start=len(leading)
                ids=leading+[config['image_token_id']]*64+tokenizer.encode('<|vision_end|>'+content[1]['text']+suffix);positions,current=image_positions(len(ids),start,64,8)
            assert ids==case['prompt_ids'] and len(ids)==3968;embeddings=io.embedding(ids);hidden=[io.visual_token(visual,t-start) if visual is not None and start<=t<start+64 else io.row(embeddings,t) for t in range(len(ids))]
            stem='sequence-'+str(sequence);session.reset();reset=state(stem+'-reset');assert same(reset,report['empty_state']);prediction=decoder.prefill_sequence(hidden,[positions],[3968]);ordinary_prefill=state(stem+'-ordinary-prefill');ordinary=[];slot=current
            for step in range(128):
                ordinary.append(prediction[0])
                if step+1<128:prediction=decoder.step(prediction,[1],[[slot]*3]);slot+=1
            ordinary_final=state(stem+'-ordinary-final');assert ordinary==case['generated_ids_by_mode']['ordinary'] and decoder.slots==[4095]
            session.reset();first=[];session.prefill(hidden,[positions],[3968],[current],on_first_token=lambda values:first.append(list(values)));candidate_prefill=state(stem+'-candidate-prefill');assert same(ordinary_prefill,candidate_prefill)
            callbacks=[];result=session.generate(128,on_token=lambda row,token,n:callbacks.append([row,token,n]));generated=list(result['generated_token_ids'][0]);candidate_final=state(stem+'-candidate-final')
            record={'sequence':sequence,'case_index':index,'key':case['key'],'request_sha256':case['request_sha256'],'image_sha256':case['image_sha256'],'prompt_ids':ids,'ordinary_ids':ordinary,'candidate_ids':generated,'reset_state':reset,'ordinary_prefill':ordinary_prefill,'candidate_prefill':candidate_prefill,'ordinary_final':ordinary_final,'candidate_final':candidate_final,'prefill_first_tokens':first,'callbacks':callbacks,'sampled_tokens_including_stop':result['sampled_tokens_including_stop'],'stop_token_ids':result['stop_token_ids'],'speculative_metrics':result['speculative_metrics']}
            report['cases'].append(record);save();assert ordinary==generated==case['generated_ids_by_mode']['mtp'] and len(generated)==128 and same(ordinary_final,candidate_final) and callbacks==[[0,t,i+1] for i,t in enumerate(generated)] and first==[[generated[0]]] and result['speculative_metrics']['mtp_prefill_calls']==249
            print('MTP_PREFILL16_FULL4K_CASE_PASS',sequence,case['key'],flush=True);del hidden,embeddings,visual
        session.reset();report['final_reset']=state('final-reset');assert same(report['empty_state'],report['final_reset']);report['memory_after']=session.memory_info();report['pool_after']=json.loads(io.memory_json());report['status']='mtp_prefill16_full4k_collected_needs_replay';save()
        report['local_replay']=replay_report(report,lambda name:(out/name).read_bytes(),cases);save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if session is not None:session.close()
        if vision is not None:vision.close()
        if decoder is not None:decoder.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
