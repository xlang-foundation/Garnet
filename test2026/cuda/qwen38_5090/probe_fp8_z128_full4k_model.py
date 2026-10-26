"""Actual T128 target prefill with retained T4 verification and T16 draft prefill; exact retained full4K gates."""
from pathlib import Path
import base64,hashlib,json,os,sys,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    root,out=map(Path,sys.argv[1:3]);mode='default';selection_off=len(sys.argv)==4;assert len(sys.argv) in (3,4) and (not selection_off or sys.argv[3]=='--selection-off') and not out.exists()
    m=root/'Garnet/test2026/cuda/qwen38_5090';repo=root/'Garnet'
    admission=json.loads((m/'evidence/fp8-z128-full4k-model-preparation.json').read_bytes())
    library=sha((root/'build/bin/garnet.dll').read_bytes());assert library==admission['candidate_library_sha256']
    for path,digest in admission['installed_files'].items():assert sha((root/path).read_bytes())==digest,('Model input/source drift',path)
    for path,digest in admission['runtime'].items():
        if path!='garnet.dll':assert sha((root/'build/bin'/path).read_bytes())==digest
    flag='GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES';assert os.environ.get(flag)==('1' if mode=='tiled' else '0')
    sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(repo))
    import garnet
    route=garnet.bf16_low_batch_tiled_scores_enabled();assert type(route) is bool and route==(mode=='tiled')
    for flag in ('GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M32_CANONICAL','GARNET_PARALLEL_BATCH_TOP1'):assert os.environ.get(flag)=='1'
    assert os.environ.get('GARNET_FP8_GDN_M128_CANONICAL')==('0' if selection_off else '1')
    assert os.environ.get('GARNET_FP8_GDN_Z128_CANONICAL')==('0' if selection_off else '1')
    parent='mtp-prefill16-full4k-admission-attempt1';baseline_proof=json.loads((root/'results'/(parent+'-verified.json')).read_bytes())
    assert baseline_proof['status']=='mtp_prefill16_full4k_target_state_replayed' and baseline_proof['archive_sha256']==admission['baseline_archive_sha256']
    archive=root/'results'/(parent+'-evidence.zip');assert sha(archive.read_bytes())==admission['baseline_archive_sha256']
    with zipfile.ZipFile(archive) as z:
        reference=json.loads(z.read('results/'+parent+'/native/report.json'));reference_files={v.filename.removeprefix('results/'+parent+'/native/'):z.read(v) for v in z.infolist() if v.filename.startswith('results/'+parent+'/native/') and v.filename.endswith('.f32')}
    cases_path=m/'evidence/native-host-stage-cases.json';p=json.loads(cases_path.read_bytes());cases=p['cases'];assert len(cases)==4 and p['library_sha256']==admission['runtime']['garnet.dll']
    sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,str(repo))
    import garnet
    from xModel.qwen38.vl_27b.model import create_cuda_bf16_prefill128_study_decoder,create_cuda_vision,prepare_cuda_image
    from xModel.qwen35.vl_0_8b.cuda_mtp_prefill16_target128_study import CUDAMTPPrefill16Target128StudySession
    from test2026.cuda.qwen38_5090.native_prompt import image_positions
    from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
    from test2026.cuda.qwen38_5090.mtp_prefill16_full4k import ORDER,replay_report
    from test2026.cuda.qwen38_5090.tiled_scores_model import same_record
    from test2026.cuda.qwen38_5090.fp8_z128_full4k_model import replay_full4k_model as replay_prefill128_model,validate_targets,validate_selection
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes());out.mkdir();decoder=vision=session=None
    report={'fp8_z128_policy':os.environ['GARNET_FP8_GDN_Z128_CANONICAL'],'fp8_m128_policy':os.environ['GARNET_FP8_GDN_M128_CANONICAL'],'status':'constructing','physical_batch':1,'capacity':4096,'library_sha256':library,'baseline_full4k_archive_sha256':baseline_proof['archive_sha256'],'tiled_scores_enabled':bool(route),'cases_sha256':sha(cases_path.read_bytes()),'cases':[],'ordinary_observations':[],'target_prefill_chunk':128,'target_decode_chunk':4,'draft_prefill_chunk':16,'target128_enabled':True,'quality_qualified':False,'performance_admitted':False,'scope':'Separate complete T128 target forward on CPU-qualified native candidate versus retained1a1d T32 full4K baseline. T4 target verification and T16 draft prefill preserved. Six3968+128text/image/reset-repeat cases require exact ordinary/MTP IDs, complete target state hashes and raw vectors. Save actual ordinary IDs and prefill/final witnesses before any equality assertion. Original old guards and gates unchanged. No broad805quality, HTTP, timing, throughput or2x admission.'}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    save()
    try:
        decoder=create_cuda_bf16_prefill128_study_decoder(str(checkpoint),config);decoder.enable_cuda_graph(True);io=decoder.io
        report['fp8_target_graphs']={str(t):json.loads(program.graph.stats_json()) for t,program in decoder.programs.items()};save()
        validate_targets(report['fp8_target_graphs'],report['fp8_m128_policy'])
        if selection_off:
            report['status']='fp8_m128_selection_off_constructed';save();return 0
        vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
        session=CUDAMTPPrefill16Target128StudySession(decoder,str(checkpoint),[],3,head_candidates=0,prefix_replay=True,draft_chunk_size=4,draft_builder_workspace_bytes=128<<20)
        tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));width=decoder.programs[4].state_width
        report['memory_after_construction']=session.memory_info();report['pool_before']=json.loads(io.memory_json());report['target_program_chunks']=sorted(decoder.programs);report['prefill_helper']=json.loads(decoder.bf16_prefill_commit.stats_json());report['decode_helper']=json.loads(decoder.delta_commit.stats_json());assert decoder.bf16_kv_prefill128_study is True and sorted(decoder.programs)==[4,128];report['fp8_auxiliary_graphs']={'draft4':report['memory_after_construction']['mtp_graph'],'draft16':report['memory_after_construction']['prefill16_graph'],'prefill_norm':report['memory_after_construction']['wide_target_norm_graph'],'vision':json.loads(vision.graph.stats_json())};save();report['fp8_selection']=validate_selection(report);report['status']='cases_running';save()
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
            ordinary_final=state(stem+'-ordinary-final');report['ordinary_observations'].append({'sequence':sequence,'case_index':index,'key':case['key'],'request_sha256':case['request_sha256'],'image_sha256':case['image_sha256'],'prompt_ids':ids,'ordinary_ids':ordinary,'reset_state':reset,'ordinary_prefill':ordinary_prefill,'ordinary_final':ordinary_final});save();assert ordinary==case['generated_ids_by_mode']['ordinary'] and decoder.slots==[4095]
            session.reset();first=[];session.prefill(hidden,[positions],[3968],[current],on_first_token=lambda values:first.append(list(values)));candidate_prefill=state(stem+'-candidate-prefill');assert same(ordinary_prefill,candidate_prefill)
            callbacks=[];result=session.generate(128,on_token=lambda row,token,n:callbacks.append([row,token,n]));generated=list(result['generated_token_ids'][0]);candidate_final=state(stem+'-candidate-final')
            record={'sequence':sequence,'case_index':index,'key':case['key'],'request_sha256':case['request_sha256'],'image_sha256':case['image_sha256'],'prompt_ids':ids,'ordinary_ids':ordinary,'candidate_ids':generated,'reset_state':reset,'ordinary_prefill':ordinary_prefill,'candidate_prefill':candidate_prefill,'ordinary_final':ordinary_final,'candidate_final':candidate_final,'prefill_first_tokens':first,'callbacks':callbacks,'sampled_tokens_including_stop':result['sampled_tokens_including_stop'],'stop_token_ids':result['stop_token_ids'],'speculative_metrics':result['speculative_metrics']}
            report['cases'].append(record);save()
            same_record(record,reference['cases'][sequence],lambda name:(out/name).read_bytes(),reference_files.__getitem__)
            assert ordinary==generated==case['generated_ids_by_mode']['mtp'] and len(generated)==128 and same(ordinary_final,candidate_final) and callbacks==[[0,t,i+1] for i,t in enumerate(generated)] and first==[[generated[0]]] and result['speculative_metrics']['mtp_prefill_calls']==249
            print('PREFILL128_FULL4K_CASE_PASS',sequence,case['key'],flush=True);del hidden,embeddings,visual
        session.reset();report['final_reset']=state('final-reset');assert same(report['empty_state'],report['final_reset']);report['memory_after']=session.memory_info();report['pool_after']=json.loads(io.memory_json());report['status']='mtp_prefill16_full4k_collected_needs_replay';save()
        report['local_replay']=replay_prefill128_model(report,lambda name:(out/name).read_bytes(),reference,reference_files.__getitem__,cases);save()
    except BaseException as error:
        report.update(status='failed',error=repr(error));save()
        if 'buildSerializedNetwork failed' in report['error']:
            try:
                from test2026.cuda.qwen38_5090.fp8_z128_native_audit_bridge import observe
                target=root/'results/fp8-z128-native-package-audit-attempt1'/('full4k-model-attempt1-'+('off' if selection_off else 'native')+'-context.json')
                report['initialization_audit_decoder_available']=decoder is not None
                report['initialization_context_audit']=observe(root,target)
                raw=target.read_bytes();copy=out/'initialization-context-audit.json';assert not copy.exists();copy.write_bytes(raw);save()
            except BaseException as audit_error:report['initialization_audit_error']=repr(audit_error);save()
        raise
    finally:
        if session is not None:session.close()
        if vision is not None:vision.close()
        if decoder is not None:decoder.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
