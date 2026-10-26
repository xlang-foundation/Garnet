"""Fresh B4 full-cache/BF16-value reference for closed actual paged B8 inputs.

Unrun. Explicit owner/preparation requires actual closed two-machine B8 proof.
No old parser/profile guard is widened and no timing is measured here.
"""
from pathlib import Path
import base64,hashlib,json,os,sys,zipfile
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from xModel.qwen38.vl_27b.model import create_cuda_vision,prepare_cuda_image
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.paged_prompt_rows import PagedPromptRows
from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
from test2026.cuda.qwen38_5090.bf16_batch8_external_reference import capture_exact,case_exact,replay_reference

PARENT='paged-batch8-model-preservation-attempt1'
sha=lambda raw:hashlib.sha256(raw).hexdigest()


def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);assert not out.exists()
    methods=Path(__file__).resolve().parent
    preparation=json.loads((methods/'evidence/bf16-batch8-external-reference-preparation.json').read_bytes())
    assert preparation['status']=='bf16_batch8_external_reference_prepared_gpu_unrun'
    library=sha((root/'build/bin/garnet.dll').read_bytes())
    assert library==preparation['runtime']['garnet.dll']
    for n,h in preparation['files'].items():assert sha((root/n).read_bytes())==h
    proof_path=root/'results'/(PARENT+'-verified.json');q=json.loads(proof_path.read_bytes());parent=preparation['parents'][PARENT]
    archive=root/'results'/(PARENT+'-evidence.zip')
    assert sha(proof_path.read_bytes())==parent['proof_sha256'] and sha(archive.read_bytes())==q['archive_sha256']==parent['archive_sha256']
    assert q['status']=='paged_batch8_model_preservation_replayed' and q['preserved_model']['all_retained_states_vectors_tokens_exact'] is True
    closed=json.loads((root/'results'/PARENT/'workflow.json').read_bytes())
    assert type(closed['exit_code']) is int and closed['exit_code']==0 and closed['cleanup_finished'] is closed['library_unchanged'] is True
    assert closed['source_sha256']==closed['source_after_sha256'] and closed['runtime_dependencies']==closed['runtime_dependencies_after']==preparation['runtime']
    with zipfile.ZipFile(archive) as z:
        prefix='results/'+PARENT+'/model/';baseline=json.loads(z.read(prefix+'report.json'))
        baseline_vectors={n.removeprefix(prefix):z.read(n) for n in z.namelist() if n.startswith(prefix) and n.endswith('.f32')}
    assert baseline['library_sha256']==library and len(baseline['cases'])==9
    bodies={}
    for key in ('s3-text-c1-r0-greedy','s3-image-c1-r0-greedy','red','green'):
        raw=(root/'results/native-mtp3-http'/(key+'-request.json')).read_bytes();bodies[key]=(raw,json.loads(raw))
    with zipfile.ZipFile(methods/'evidence/fixed-length-4096-corpus.zip') as z:
        raw=z.read('manifest.json');assert sha(raw)=='25808fff53f326b6a1a970bb389173b2a54c1d1189248d40122fcdb959cb6987'
        for row in json.loads(raw)['rows']:
            wire=z.read(row['request_file']);assert sha(wire)==row['request_sha256'];bodies[row['key']]=(wire,json.loads(wire))
    for flag in ('GARNET_FP8_GDN_M32_CANONICAL','GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_PARALLEL_BATCH_TOP1'):assert os.environ.get(flag)=='1'
    out.mkdir();path=out/'report.json';checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes())
    report={'status':'constructing_reference','physical_batch':4,'capacity':4096,'prefill_chunk':32,'decode_chunk':4,
            'bf16_kv_float_reference':True,'compact_commit_used':False,'mtp_executed':False,'http_executed':False,
            'library_sha256':library,'baseline_archive_sha256':q['archive_sha256'],'baseline_proof_sha256':sha(proof_path.read_bytes()),
            'empty_state':{},'final_reset':{},'cases':[],'performance_admitted':False,
            'scope':'Fresh ordinary existing B4/T32/T4 full-FP32-cache with explicit BF16-value roundtrip reference. Same actual closed B8 inputs split into two ordered four-row groups; nine original/rotated/reset cases become18fresh cases. Complete expanded logical states and raw hidden/normalized vectors/all IDs must exactly match B8. Fixed-count8short/128full4K ignores EOS like the original B8 oracle. Existing core guards/arithmetic unchanged; shared target projections/linear operators are not an independent framework oracle. No MTP/805quality/HTTP/timing/2x admission.'}
    def save():path.write_bytes(json.dumps(report,indent=2).encode())
    decoder=vision=None;save()
    try:
        decoder=CUDAPackedDecoder(str(checkpoint),config['text_config'],4,4096,32,4,flat_concat=True,
            external_compressed_weights=True,retain_linear_inputs=True,deferred_state_commit=True,
            host_embedding_lookup=True,joined_mlp_projections=True,shared_output_storage=True,
            register_delta_128=True,bf16_kv_float_reference=True)
        assert decoder.delta_commit is decoder.bf16_prefill_commit is None and not decoder.bf16_kv_state and not decoder.kv_delta_state
        assert all(p.bf16_kv_float_reference is True and p.delta_layout is None and p.state_width==173932544 for p in decoder.programs.values())
        decoder.io.enable_nvfp4_head(True);decoder.enable_cuda_graph(True);io=decoder.io
        pool=json.loads(io.memory_json());assert pool['embedding_mode']=='host_mapped_bf16' and pool['head_precision']=='modelopt_nvfp4_w4a4'
        assert (pool['compressed_projection_pool_entries'],pool['compressed_nvfp4_entries'],pool['compressed_fp8_entries'])==(336,128,208)
        report['pool_before']=pool;report['decoder_memory_after_construction']=decoder.memory_info();save()
        vision=create_cuda_vision(str(checkpoint),config,16,16);vision.graph.enable_cuda_graph(True)
        tokenizer=garnet.tokenizer();tokenizer.open(str(checkpoint));stops=[tokenizer.token_id('<|im_end|>'),tokenizer.token_id('<|endoftext|>')]
        suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        def snapshot(label):
            assert decoder.state_transaction is None and not decoder.poisoned
            logical=logical_state_witness(io,decoder.state,decoder.programs[4].layout,4,173932544,decoder.slots,packed_bf16=False)
            vectors={}
            for kind,value in (('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))):
                raw=bytes(io.finite_f32_bytes(value));assert len(raw)==81920;filename=label+'-'+kind+'.f32';(out/filename).write_bytes(raw)
                vectors[kind]={'file':filename,'bytes':len(raw),'sha256':sha(raw)}
            return {'logical':logical,'vectors':vectors}
        read=lambda name:(out/name).read_bytes()
        for half in (0,1):
            decoder.reset();state=snapshot('empty-half'+str(half));report['empty_state'][str(half)]=state;save()
            capture_exact(state,baseline['empty_state'],list(range(half*4,half*4+4)),read,baseline_vectors.__getitem__)
        report['status']='reference_cases_running';save()
        for case_index,original in enumerate(baseline['cases']):
            for half in (0,1):
                physical_rows=list(range(half*4,half*4+4));inputs=[original['inputs'][i] for i in physical_rows]
                stem='case'+str(case_index)+'-half'+str(half);decoder.reset();reset=snapshot(stem+'-reset');prepared=[]
                for physical,row in enumerate(inputs):
                    wire,body=bodies[row['key']];assert sha(wire)==row['request_sha256'];content=body['messages'][0]['content'];visual=None;start=-1
                    if isinstance(content,str):
                        ids=tokenizer.encode('<|im_start|>user\n'+content+suffix);positions=[[i]*3 for i in range(len(ids))];current=len(ids);assert row['image_sha256'] is None
                    else:
                        raw=base64.b64decode(content[0]['image_url']['url'].split(',',1)[1],validate=True);assert sha(raw)==row['image_sha256']
                        image=out/(stem+'-row'+str(physical)+'.jpg');image.write_bytes(raw);patches=prepare_cuda_image(io,str(checkpoint),str(image));assert [patches[1],patches[2]]==[16,16]
                        visual=io.clone(vision.encode_patches(patches[0]));leading=tokenizer.encode('<|im_start|>user\n<|vision_start|>');start=len(leading)
                        ids=leading+[config['image_token_id']]*64+tokenizer.encode('<|vision_end|>'+content[1]['text']+suffix);positions,current=image_positions(len(ids),start,64,8)
                    assert ids==row['prompt_ids'];prepared.append({'ids':ids,'positions':positions,'current':current,'visual':visual,'start':start})
                lengths=[len(v['ids']) for v in prepared];count=max(lengths);limit=original['actual_limit'];assert limit in (8,128) and all(n+limit<=4096 for n in lengths)
                hidden=PagedPromptRows(io,[v['ids'] for v in prepared],[v['visual'] for v in prepared],[v['start'] for v in prepared],page_size=128)
                positions=[v['positions']+[[0,0,0]]*(count-len(v['ids'])) for v in prepared];current=[v['current'] for v in prepared]
                try:prediction=decoder.prefill_sequence(hidden,positions,lengths);cache=hidden.stats()
                finally:hidden.close()
                prefill=snapshot(stem+'-prefill');generated=[[] for _ in range(4)]
                for step in range(limit):
                    assert len(prediction)==4
                    for row,token in enumerate(prediction):assert type(token) is int and 0<=token<248320;generated[row].append(token)
                    if step+1<limit:prediction=decoder.step(prediction,[1]*4,[[v]*3 for v in current]);current=[v+1 for v in current]
                final=snapshot(stem+'-final');assert decoder.slots==[n+limit-1 for n in lengths]
                record={'baseline_case_index':case_index,'half':half,'physical_rows':physical_rows,
                        **{key:original[key] for key in ('group','repeat','order','actual_limit')},'inputs':inputs,'generated_ids':generated,
                        'text':[tokenizer.decode([v for v in row if v not in stops]) for row in generated],
                        'reset_state':reset,'prefill_state':prefill,'final_state':final,'prompt_cache':cache,
                        'all_reference_rows_exact':False,'performance_admitted':False,'decoder_memory_after_case':decoder.memory_info()}
                report['cases'].append(record);save();case_exact(record,original,case_index,half,read,baseline_vectors.__getitem__)
                record['all_reference_rows_exact']=True;save();print('B8_EXTERNAL_B4_REFERENCE_CASE_PASS',case_index,half,flush=True)
                del hidden,prepared,visual
        for half in (0,1):
            decoder.reset();state=snapshot('final-reset-half'+str(half));report['final_reset'][str(half)]=state;save()
            capture_exact(state,baseline['final_reset'],list(range(half*4,half*4+4)),read,baseline_vectors.__getitem__)
        report['status']='bf16_batch8_external_reference_collected_needs_replay';save()
        report['local_replay']=replay_reference(report,read,baseline,baseline_vectors.__getitem__);save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if vision is not None:vision.close()
        if decoder is not None:decoder.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
