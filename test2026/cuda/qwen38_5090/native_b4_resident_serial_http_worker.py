"""Candidate B4 resident controls and serial graph reuse; fresh correctness/performance admission required."""
from pathlib import Path
import base64,hashlib,json,os,sys,time
root=Path(sys.argv[1]);sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')]
from test2026.cuda.qwen38_5090.b4_rowwise_http_admission import require
admission=require(root)
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen35.vl_0_8b.cuda_mtp_prefill16_b4_study import CUDAMTPPrefill16B4StudySession
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.rowwise_k_norm_program import RowwiseKDecodeNormProgram,construct_with_rowwise_k_decode_norm_program
from test2026.cuda.qwen38_5090.paged_prompt_rows import PagedPromptRows
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.native_worker_io import requests,emit
from test2026.cuda.qwen38_5090.b4_serving_request import partition_jobs
from test2026.cuda.qwen38_5090.b4_mtp_public_stream import PublicBatchStream

checkpoint=str(root/'models/Qwen3.8-27B-NVFP4');config=json.loads(Path(checkpoint,'config.json').read_bytes())
decoder=vision=session=None
try:
    decoder=construct_with_rowwise_k_decode_norm_program(CUDAPackedDecoder,checkpoint,config['text_config'],4,4096,32,4,
        flat_concat=True,external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,
        host_embedding_lookup=True,joined_mlp_projections=True,shared_output_storage=True,register_delta_128=True,bf16_kv_state=True)
    # Existing production policy requires all three for packed shared output.
    decoder.resident_controls=True
    decoder.serial_target_graph_reuse=True
    decoder.serial_graph_reuse=True
    decoder.io.enable_nvfp4_head(True);decoder.enable_cuda_graph(True);io=decoder.io
    assert isinstance(decoder.programs[4],RowwiseKDecodeNormProgram) and sorted(decoder.programs)==[4,32]
    pool=json.loads(io.memory_json());assert pool['embedding_mode']=='host_mapped_bf16' and pool['head_precision']=='modelopt_nvfp4_w4a4'
    assert (pool['compressed_projection_pool_entries'],pool['compressed_nvfp4_entries'],pool['compressed_fp8_entries'])==(336,128,208)
    counts={'native_fp8_m128_canonical_projections':48,'native_fp8_z128_canonical_projections':0,'native_fp8_z128_reconstructed_projections':48,'native_fp8_kv128_canonical_projections':32,'native_fp8_q128_canonical_projections':16,'native_fp8_o128_canonical_projections':16,'native_full_chunk_stage_witnesses':0}
    graphs={str(t):json.loads(program.graph.stats_json()) for t,program in decoder.programs.items()}
    for t,stats in graphs.items():
        for key,count in counts.items():assert type(stats[key]) is int and stats[key]==(count if t=='32' else 0)
    vision=create_cuda_vision(checkpoint,config,16,16);vision.graph.enable_cuda_graph(True)
    vision_stats=json.loads(vision.graph.stats_json());assert all(vision_stats[key]==0 for key in counts)
    session=CUDAMTPPrefill16B4StudySession(decoder,checkpoint,[],3,head_candidates=0,prefix_replay=True,
        draft_chunk_size=4,draft_builder_workspace_bytes=256<<20,draft_prefill_builder_workspace_bytes=256<<20)
    assert session.verifier.reuse_verified_head is False
    if session.prefill_norm is not None:session.prefill_norm.enable_serial_graph_reuse(True)
    def graph_snapshot():
        objects={**{'target'+str(t):program.graph for t,program in decoder.programs.items()},
                 'decoder_norm':decoder.head,'verifier_norm':session.verifier.graph,
                 'prefix_commit':session.verifier.replay_graph,'draft4':session.program.graph,
                 'draft16':session.prefill16_program.graph,'wide_norm':session.prefill_norm}
        return {name:json.loads(graph.stats_json()) for name,graph in objects.items() if graph is not None}
    initial_graphs=graph_snapshot()
    assert all(value['cuda_graph_requested'] is True and value['serial_graph_reuse_requested'] is True for value in initial_graphs.values())
    memory=session.memory_info();assert (memory['physical_batch'],memory['target_prefill_chunk'],memory['draft_prefill_chunk'],memory['draft_decode_chunk'])==(4,32,16,4)
    for graph in (memory['mtp_graph'],memory['prefill16_graph'],memory['wide_target_norm_graph']):assert all(graph[key]==0 for key in counts)
    assert memory['mtp_graph']['builder_workspace_limit_bytes']==memory['prefill16_graph']['builder_workspace_limit_bytes']==256<<20
    tokenizer=garnet.tokenizer();tokenizer.open(checkpoint);stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
    suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    emit(dict(event='ready',protocol='study_batch_v1',batch=4,capacity=4096,prefill_chunk=32,chunk=4,
              generation_mode='verified_mtp',mtp_depth=3,library_sha256=admission['library_sha256'],
              source_admission=admission,physical_routes=graphs,mtp_memory=memory,
              mtp_policy='homogeneous_limit_and_ignore_eos_subgroups_private_padding_to_four',
              padding_policy='duplicate_first_real_row_no_public_dummy_events',frame_max_bytes=16*1024*1024,
              explicit_ignore_eos_supported=True,precision='mixed NVFP4/FP8; BF16 vision/cache; checkpoint NVFP4 head',
              research_only=True,adapted805_HTTP_or_performance_qualified=False,
              backend_policy=dict(resident_controls=True,serial_target_graph_reuse=True,serial_draft_graph_reuse=True,
                                  accepted_head_reuse=False),initial_graphs=initial_graphs,
              backend_policy_correctness_or_performance_qualified=False))
    for frame in requests(max_frame_bytes=16*1024*1024):
        groups=partition_jobs(frame.get('batch'))
        for jobs in groups:
            snapshot_flags=[job['body'].get('study_graph_snapshot',False) for job in jobs]
            if any(type(value) is not bool for value in snapshot_flags):raise ValueError('study_graph_snapshot requires a boolean')
            started=time.monotonic();session.reset();prepared=[]
            for row,job in enumerate(jobs):
                content=job['body']['messages'][0]['content'];visual=None;start=-1;image_hash=None
                if isinstance(content,str):
                    ids=tokenizer.encode('<|im_start|>user\n'+content+suffix);positions=[[i]*3 for i in range(len(ids))];current=len(ids)
                else:
                    data=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True);assert data[:2]==b'\xff\xd8'
                    image_hash=hashlib.sha256(data).hexdigest();image=root/'tmp'/('b4-rowwise-stream-'+str(row)+'.jpg');image.write_bytes(data)
                    leading=tokenizer.encode('<|im_start|>user\n<|vision_start|>');start=len(leading)
                    ids=leading+[config['image_token_id']]*64+tokenizer.encode('<|vision_end|>'+content[1]['text']+suffix)
                    positions,current=image_positions(len(ids),start,64,8)
                assert ids and len(ids)+job['body']['max_tokens']<=4096
                if not isinstance(content,str):
                    patches=prepare_cuda_image(io,checkpoint,str(image));assert [patches[1],patches[2]]==[16,16]
                    visual=io.clone(vision.encode_patches(patches[0]))
                prepared.append(dict(ids=ids,positions=positions,current=current,visual=visual,start=start,image_hash=image_hash))
            physical=prepared+[prepared[0]]*(4-len(prepared));n=max(len(v['ids']) for v in physical)
            positions=[v['positions']+[[0,0,0]]*(n-len(v['ids'])) for v in physical]
            public=PublicBatchStream(jobs,[v['ids'] for v in prepared],stops,tokenizer.decode,emit,[v['image_hash'] for v in prepared],started)
            session.stops=set(public.stop_ids)
            hidden=PagedPromptRows(io,[v['ids'] for v in physical],[v['visual'] for v in physical],[v['start'] for v in physical],page_size=128)
            def first(prediction):
                for row,token in enumerate(prediction):public.notify(row,token,1)
            try:session.prefill(hidden,positions,[len(v['ids']) for v in physical],[v['current'] for v in physical],on_first_token=first)
            finally:hidden.close()
            public.complete(session.generate(jobs[0]['body']['max_tokens'],on_token=public.notify))
            # Diagnostic snapshots are outside public token delivery, opt-in,
            # and absent from future clean timing request bodies.
            if any(snapshot_flags):
                emit(dict(event='study_graph_snapshot',request_ids=[job['id'] for job in jobs],
                          real_rows=len(jobs),physical_batch=4,graphs=graph_snapshot(),
                          io=json.loads(io.memory_json()),CUDA_kernel_elapsed_claimed=False,
                          clean_throughput_result_claimed=False))
finally:
    if session is not None:session.close()
    if vision is not None:vision.close()
    if decoder is not None:decoder.close()
