"""Capture existing layer0 retained projections at physical B1T4/B4T4."""
from pathlib import Path
import hashlib,json,os,sys
sha=lambda b:hashlib.sha256(b).hexdigest()
WEIGHT_NAMES=tuple('model.language_model.layers.0.'+s for s in ('mlp.gate_up_proj.weight','mlp.down_proj.weight','linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight','linear_attn.out_proj.weight'))
def norm_forward(hidden):return hidden*T.unary_op('qwen35_rms_norm',weight_name='model.language_model.layers.0.input_layernorm.weight',eps=1.e-6)
def layer_forward(hidden,history,state,active,positions,slots,*weights):
    compressed={}
    for index in range(len(WEIGHT_NAMES)):
        values=[]
        for slot in WEIGHT_SLOTS[index]:values.append(weights[slot])
        compressed[WEIGHT_NAMES[index]]=values
    return Qwen35CUDAChunk(hidden,history,state,active,positions,slots,CONFIG,0,'linear_attention',fused_delta=True,fused_attention=True,retain_linear_inputs=True,register_delta_128=True,external_compressed_weights=compressed,joined_mlp_projections=True)

def main():
    global T,Qwen35CUDAChunk,CONFIG,WEIGHT_SLOTS
    root,out,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];assert mode in ('reference','candidate') and not out.exists();m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/decode-projection-attempt3-preparation.json').read_bytes());batch=1 if mode=='reference' else 4
    for n,h in p[mode+'_installed_files'].items():assert sha((root/n).read_bytes())==h
    assert sha((root/'build/bin/garnet.dll').read_bytes())==p[mode+'_library_sha256']
    assert {n:os.environ.get(n) for n in p['flags']}=={**p['flags'],'GARNET_FP8_BATCH4_PREFILL_CANONICAL':'0' if mode=='reference' else '1'}
    for n,h in p['runtime'].items():assert sha((root/'build/bin'/n).read_bytes())==(p[mode+'_library_sha256'] if n=='garnet.dll' else h)
    for n,h in p['checkpoints'].items():assert sha((root/n).read_bytes())==h
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')];import garnet,tensor
    from xModel.qwen35.vl_0_8b.decoder_layer import T
    from xModel.qwen35.vl_0_8b.cuda_chunk import Qwen35CUDAChunk
    from test2026.cuda.qwen38_5090.decode_projection_diagnostic import STAGES,OFFSETS,TOTAL
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';CONFIG=json.loads((checkpoint/'config.json').read_bytes())['text_config'];assert CONFIG['rms_norm_eps']==1.e-6 and CONFIG['hidden_size']==5120
    out.mkdir();report=dict(status='constructing',mode=mode,physical_batch=batch,chunk=4,library_sha256=p[mode+'_library_sha256'],token_inputs=p['token_inputs'],cases=[],full_model_or_performance_qualified=False,scope=p['scope']);io=graph=normalizer=None
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        io=garnet.cuda_tensor_io();io.open(str(checkpoint));io.enable_nvfp4_head(True);io.enable_host_embedding(True);weights=[];symbols=[];WEIGHT_SLOTS=[];report['weight_metadata']={}
        for name in WEIGHT_NAMES:
            info=json.loads(io.compressed_weight_info(name));report['weight_metadata'][name]=info;indices=[];buffers=[('packed',info['packed_shape'],io.compressed_weight(name))]
            if info['scale_bytes']:buffers.append(('scales',info['scale_shape'],io.compressed_scales(name)))
            for role,shape,value in buffers:
                indices.append(len(weights));weights.append(value);symbols.append(tensor.input('decode_'+role+str(len(weights)),shape=shape,dtype=tensor.int32))
            WEIGHT_SLOTS.append(indices)
        shapes=[[batch,4,5120],[batch,10240,4],[batch,48,128,128],[batch,4]];inputs=[tensor.input('decode_input'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
        @T.fusion(name='qwen38_actual_decode_layer0',boundary='required')
        def actual_layer(hidden,history,state,active,*provided_weights):
            compressed={}
            for index in range(len(WEIGHT_NAMES)):
                values=[]
                for slot in WEIGHT_SLOTS[index]:values.append(provided_weights[slot])
                compressed[WEIGHT_NAMES[index]]=values
            return Qwen35CUDAChunk(hidden,history,state,active,None,None,CONFIG,0,'linear_attention',fused_delta=True,fused_attention=True,retain_linear_inputs=True,register_delta_128=True,external_compressed_weights=compressed,joined_mlp_projections=True)
        @T.fusion(name='qwen38_actual_decode_norm',boundary='required')
        def actual_norm(hidden):return hidden*T.unary_op('qwen35_rms_norm',weight_name='model.language_model.layers.0.input_layernorm.weight',eps=1.e-6)
        graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(256<<20);graph.build(actual_layer(*inputs,*symbols),inputs+symbols,str(checkpoint));graph.enable_cuda_graph(True)
        norm_symbol=tensor.input('decode_norm',shape=shapes[0],dtype=tensor.float32);normalizer=garnet.compiled_graph();normalizer.set_builder_workspace_bytes(128<<20);normalizer.build(actual_norm(norm_symbol),[norm_symbol],str(checkpoint));normalizer.enable_cuda_graph(True)
        controls=[io.zeros(s) for s in shapes];hidden,history,state,active=controls;io.update(active,[1,0,0,0]*batch);report['graph_before']=json.loads(graph.stats_json());report['norm_graph_before']=json.loads(normalizer.stats_json());save()
        assert report['graph_before']['native_fp8_projection_weights']==3 and report['graph_before']['native_nvfp4_projection_weights']==2 and report['graph_before']['external_compressed_weight_inputs']==5
        for stats in (report['graph_before'],report['norm_graph_before']):
            for key in ('native_fp8_m128_canonical_projections','native_fp8_z128_canonical_projections','native_fp8_z128_reconstructed_projections','native_fp8_kv128_canonical_projections','native_fp8_q128_canonical_projections','native_fp8_o128_canonical_projections','native_full_chunk_stage_witnesses'):assert stats[key]==0
        def capture(value,name,words):
            raw=bytes(io.finite_f32_bytes(value));assert len(raw)==batch*words*4;file=name+'.f32';assert not (out/file).exists();(out/file).write_bytes(raw);return dict(file=file,bytes=len(raw),sha256=sha(raw))
        groups=[[i] for i in range(4)] if batch==1 else [[0,1,2,3]];report['status']='capturing_actual_decode_projections';save()
        for step,all_tokens in enumerate(p['token_inputs']):
            for group in groups:
                tokens=[all_tokens[i] for i in group];io.zero(hidden);io.write_chunk_token(io.embedding(tokens),hidden,0);io.zero(history);io.zero(state);stem='step'+str(step)+'-row'+str(group[0]);entry=dict(step=step,physical_rows=group,tokens=tokens,repeats=[],input_after_sha256=[]);report['cases'].append(entry);entry['input']=capture(hidden,stem+'-input',4*5120);save()
                for repeat in range(2):
                    label=stem+'-repeat'+str(repeat);normalized=normalizer.run_device([hidden]);packed=graph.run_device(controls+weights);arrays={'input':entry['input'],'normalized':capture(normalized,label+'-normalized',4*5120)}
                    for key,offset in OFFSETS.items():
                        part=io.zeros([batch,4*STAGES[key]]);io.slice_into(packed,offset,part);arrays[key]=capture(part,label+'-'+key,4*STAGES[key])
                    after=bytes(io.finite_f32_bytes(hidden));entry['input_after_sha256'].append(sha(after));assert sha(after)==entry['input']['sha256'];entry['repeats'].append(arrays);save()
                print('DECODE_PROJECTION_CAPTURED',mode,step,group[0],flush=True)
        report.update(status='actual_decode_projection_arrays_collected',graph_after=json.loads(graph.stats_json()),norm_graph_after=json.loads(normalizer.stats_json()),io_memory=json.loads(io.memory_json()));save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if normalizer is not None:normalizer.close()
        if graph is not None:graph.close()
        if io is not None:io.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
