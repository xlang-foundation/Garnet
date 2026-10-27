"""Untimed connected full-attention component with six original compressed weights."""
from pathlib import Path
import hashlib,json,os,sys
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def full_forward(hidden,key,value,active,positions,slots,*weights):
    compressed={}
    for i in range(len(WEIGHT_NAMES)):
        values=[]
        for slot in WEIGHT_SLOTS[i]:values.append(weights[slot])
        compressed[WEIGHT_NAMES[i]]=values
    return Qwen35CUDAChunk(hidden,key,value,active,positions,slots,CONFIG,LAYER,'full_attention',fused_delta=True,fused_attention=True,external_compressed_weights=compressed,joined_mlp_projections=True)
def main():
    global T,Qwen35CUDAChunk,CONFIG,LAYER,WEIGHT_NAMES,WEIGHT_SLOTS
    root,out,mode=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];assert mode in ('reference','candidate') and not out.exists();m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/connected-fa-value-v2-preparation.json').read_bytes());batch=1 if mode=='reference' else 4
    library=sha((root/'build/bin/garnet.dll').read_bytes());assert library==p[mode+'_library_sha256']
    for name,h in p[mode+'_installed_files'].items():assert sha((root/name).read_bytes())==h
    for name,h in p['runtime'].items():assert sha((root/'build/bin'/name).read_bytes())==(library if name=='garnet.dll' else h)
    for name,h in p['checkpoints'].items():assert sha((root/name).read_bytes())==h
    assert {name:os.environ.get(name) for name in p['flags']}=={**p['flags'],'GARNET_FP8_BATCH4_PREFILL_CANONICAL':'0' if mode=='reference' else '1','GARNET_FP8_GDN_BATCH4_DECODE_CANONICAL':'0' if mode=='reference' else '1'}
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')];import garnet,tensor
    from xModel.qwen35.vl_0_8b.decoder_layer import T
    from xModel.qwen35.vl_0_8b.cuda_chunk import Qwen35CUDAChunk
    from test2026.cuda.qwen38_5090.connected_fa_value_layout import CASES
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';CONFIG=json.loads((checkpoint/'config.json').read_bytes())['text_config'];assert CONFIG['rms_norm_eps']==1.e-6 and CONFIG['hidden_size']==5120
    fixture=root/'results/connected-fa-value-fixture';f=json.loads((fixture/'manifest.json').read_bytes())
    for name,value in f['files'].items():assert sha((fixture/name).read_bytes())==value['sha256']
    source=json.loads((fixture/'actual-hidden-inputs.json').read_bytes());out.mkdir();io=graph=normalizer=None
    report=dict(status='constructing',mode=mode,physical_batch=batch,tokens=4,capacity=4096,library_sha256=library,fixture_manifest_sha256=sha((fixture/'manifest.json').read_bytes()),cache_mode='ordinary_zero_FP32_inputs_component_only',position_and_slot_controls='zero_component_controls_V_projection_independent',cases=[],internal_fused_normalization_directly_captured=False,model_or_performance_qualified=False)
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        io=garnet.cuda_tensor_io();io.open(str(checkpoint));io.enable_host_embedding(True)
        @T.fusion(name='qwen38_connected_control_norm3',boundary='required')
        def norm3(h):return h*T.unary_op('qwen35_rms_norm',weight_name='model.language_model.layers.3.input_layernorm.weight',eps=1.e-6)
        @T.fusion(name='qwen38_connected_control_norm11',boundary='required')
        def norm11(h):return h*T.unary_op('qwen35_rms_norm',weight_name='model.language_model.layers.11.input_layernorm.weight',eps=1.e-6)
        def capture(tensor,name,words):
            raw=bytes(io.finite_f32_bytes(tensor));assert len(raw)==batch*words*4;file=name+'.f32';assert not (out/file).exists();(out/file).write_bytes(raw);return dict(file=file,bytes=len(raw),sha256=sha(raw))
        cachepart=io.zeros([batch,1048576]);vpart=io.zeros([batch,256]);hiddenpart=io.zeros([batch,20480]);width=4194304
        def cache_hashes(caches):
            values=[]
            for cache in caches:
                hashes=[hashlib.sha256() for _ in range(batch)]
                for start in range(0,width,1048576):
                    io.slice_into(cache,start,cachepart)
                    for row in range(batch):hashes[row].update(bytes(io.finite_f32_bytes(io.row(cachepart,row))))
                for h in hashes:values.append(h.hexdigest())
            return values
        for label,layer,step in CASES:
            LAYER=layer;prefix='model.language_model.layers.'+str(layer);WEIGHT_NAMES=[prefix+suffix for suffix in ('.mlp.gate_up_proj.weight','.mlp.down_proj.weight','.self_attn.q_proj.weight','.self_attn.k_proj.weight','.self_attn.v_proj.weight','.self_attn.o_proj.weight')];WEIGHT_SLOTS=[];weights=[];symbols=[];metadata={}
            for name in WEIGHT_NAMES:
                info=json.loads(io.compressed_weight_info(name));metadata[name]=info;buffers=[('packed',info['packed_shape'],io.compressed_weight(name))]
                if info['scale_bytes']:buffers.append(('scales',info['scale_shape'],io.compressed_scales(name)))
                indices=[]
                for role,shape,value in buffers:indices.append(len(weights));weights.append(value);symbols.append(tensor.input('connected_'+label+role+str(len(weights)),shape=shape,dtype=tensor.int32))
                WEIGHT_SLOTS.append(indices)
            shapes=[[batch,4,5120],[batch,4,4096,256],[batch,4,4096,256],[batch,4],[batch,4,3],[batch,4]];inputs=[tensor.input('connected_'+label+str(i),shape=shape,dtype=tensor.float32) for i,shape in enumerate(shapes)]
            @T.fusion(name='qwen38_connected_full_component',boundary='required')
            def forward(*args):return full_forward(*args)
            graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(256<<20);graph.build(forward(*inputs,*symbols),inputs+symbols,str(checkpoint));graph.enable_cuda_graph(True)
            normalizer=garnet.compiled_graph();normalizer.set_builder_workspace_bytes(128<<20);symbol=tensor.input('connected_norm_input',shape=shapes[0],dtype=tensor.float32);function=norm3 if layer==3 else norm11;normalizer.build(function(symbol),[symbol],str(checkpoint));normalizer.enable_cuda_graph(True)
            controls=[io.zeros(shape) for shape in shapes];hidden,key,value,active,positions,slots=controls;io.update(active,[1,0,0,0]*batch);groups=[[r] for r in range(4)] if batch==1 else [[0,1,2,3]]
            for physical in groups:
                values=[]
                for row in physical:values.extend(source[label][row*20480:(row+1)*20480])
                io.update(hidden,values);stem=label+'-row'+str(physical[0]);record=dict(label=label,layer=layer,actual_decode_step=step,physical_rows=physical,input=capture(hidden,stem+'-input',20480),input_cache_sha256_before=cache_hashes((key,value)),graph_stats=json.loads(graph.stats_json()),norm_stats=json.loads(normalizer.stats_json()),weight_metadata=metadata,repeats=[]);report['cases'].append(record);save()
                for repeat in range(2):
                    entry=dict(repeat=repeat,unmasked_normalized=capture(normalizer.run_device([hidden]),stem+'-r'+str(repeat)+'-normalizer',20480));packed=graph.run_device(controls+weights);io.slice_into(packed,0,hiddenpart);entry['component_hidden']=capture(hiddenpart,stem+'-r'+str(repeat)+'-hidden',20480);pieces=[]
                    for head in range(4):
                        io.slice_into(packed,20480+width+head*4096*256,vpart);pieces.append(bytes(io.finite_f32_bytes(vpart)))
                    byrow=[]
                    for row in range(batch):
                        for raw in pieces:byrow.append(raw[row*1024:(row+1)*1024])
                    raw=b''.join(byrow);name=stem+'-r'+str(repeat)+'-value.f32';assert len(raw)==batch*4096 and not (out/name).exists();(out/name).write_bytes(raw);entry['component_V']=dict(file=name,bytes=len(raw),sha256=sha(raw));entry['input_after_sha256']=sha(bytes(io.finite_f32_bytes(hidden)));record['repeats'].append(entry);save()
                record['input_after_sha256']=sha(bytes(io.finite_f32_bytes(hidden)));record['input_cache_sha256_after']=cache_hashes((key,value));save();print('CONNECTED_FA_VALUE_CAPTURED',mode,label,physical,flush=True)
            normalizer.close();normalizer=None;graph.close();graph=None
        report['status']='connected_FA_value_components_collected';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if normalizer is not None:normalizer.close()
        if graph is not None:graph.close()
        if io is not None:io.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
