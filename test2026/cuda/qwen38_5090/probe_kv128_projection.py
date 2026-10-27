"""Run the actual registered Garnet witness on an admitted real layer2 hidden fixture."""
from pathlib import Path
import hashlib,json,os,struct,sys
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    root,out=map(Path,sys.argv[1:3]);m=root/'Garnet/test2026/cuda/qwen38_5090';repo=root/'Garnet';p=json.loads((m/'evidence/kv128-projection-gpu-preparation.json').read_bytes());assert not out.exists()
    for n,h in p['installed_files'].items():assert sha((root/n).read_bytes())==h
    for n,h in p['checkpoints'].items():assert sha((root/n).read_bytes())==h
    for n,h in p['runtime'].items():assert sha((root/'build/bin'/n).read_bytes())==(p['candidate_library_sha256'] if n=='garnet.dll' else h)
    assert os.environ.get('GARNET_FP8_FA3_PROJECTION_WITNESS')=='1' and os.environ.get('GARNET_MODELOPT_INIT_TRACE')=='1'
    assert os.environ.get('GARNET_FP8_FULL_ATTN_KV128_CANONICAL')=='1' and os.environ.get('GARNET_KV128_FACTORY_TRACE')=='1'
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(repo)];import garnet,tensor
    from xModel.qwen35.vl_0_8b.decoder_layer import T
    from test2026.cuda.qwen38_5090.kv128_projection_layout import STAGES,check_graph,weight_metadata
    from test2026.cuda.qwen38_5090.gdn_chunk_stage_layout import pieces
    rawinput=(m/'evidence/reconstructed-fa3-common-input.f32').read_bytes();metadata=json.loads((m/'evidence/reconstructed-fa3-common-input.json').read_bytes());assert len(rawinput)==2621440 and sha(rawinput)==metadata['sha256']==p['common_input_sha256']
    out.mkdir();report=dict(status='constructing',witness_policy='1',source_input_metadata=metadata,input_fixture_sha256=sha(rawinput),candidate_library_sha256=p['candidate_library_sha256'],cases=[],model_or_performance_qualified=False,scope=p['scope'])
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    io=graph=None;save()
    def capture(value,offset,words,label):
        parts=[];digest=hashlib.sha256()
        for start,count in pieces(words):
            part=io.zeros([1,count]);io.slice_into(value,offset+start,part);raw=bytes(io.finite_f32_bytes(part));assert len(raw)==count*4;name=label+'-'+str(start)+'.f32';target=out/name;assert not target.exists();target.write_bytes(raw);digest.update(raw);parts.append(dict(file=name,offset=start,words=count,sha256=sha(raw)))
        return dict(words=words,sha256=digest.hexdigest(),parts=parts)
    try:
        checkpoint=root/'models/Qwen3.8-27B-NVFP4';io=garnet.cuda_tensor_io();io.open(str(checkpoint));io.enable_nvfp4_head(True);io.enable_host_embedding(True)
        report['weight_metadata']={name:json.loads(io.compressed_weight_info(name)) for name in ('model.language_model.layers.3.self_attn.k_proj.weight','model.language_model.layers.3.self_attn.v_proj.weight')};weight_metadata(report['weight_metadata'])
        common=io.from_host(list(struct.unpack('<655360f',rawinput)),[1,128*5120]);report['common_input']=capture(common,0,128*5120,'common-input');save()
        for tokens in (32,128):
            symbol=tensor.input('fa3_hidden',shape=[1,tokens,5120],dtype=tensor.float32)
            @T.fusion(name='qwen38_fa3_projection_witness',boundary='required')
            def forward(hidden):return hidden*T.unary_op('qwen35_fp8_fa3_projection_witness',layer_prefix='model.language_model.layers.3',eps=1.e-6)
            graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(256<<20);graph.build(forward(symbol),[symbol],str(checkpoint));graph.enable_cuda_graph(True)
            hidden=io.zeros([1,tokens,5120]);flat=io.storage_view(hidden,[1,tokens*5120]);row=dict(tokens=tokens,repeats=[]);report['cases'].append(row);save()
            for warmup in range(3):
                io.slice_into(common,0,flat);result=graph.run_device([hidden]);boundary=io.zeros([1,1]);io.slice_into(result,0,boundary);assert len(bytes(io.finite_f32_bytes(boundary)))==4
            row['graph']=json.loads(graph.stats_json());check_graph(row['graph'],tokens);save()
            for repeat in range(2):
                sequence=[];row['repeats'].append(sequence)
                for start in range(0,128,tokens):
                    label='t'+str(tokens)+'-r'+str(repeat)+'-s'+str(start);io.slice_into(common,start*5120,flat);chunk=dict(start=start,repeat=repeat,arrays={});sequence.append(chunk);chunk['input_hidden']=capture(flat,0,tokens*5120,label+'-input');result=graph.run_device([hidden]);offset=0
                    for name,width in STAGES:chunk['arrays'][name]=capture(result,offset,tokens*width,label+'-'+name);offset+=tokens*width
                    chunk['input_hidden_after']=capture(flat,0,tokens*5120,label+'-after-input');save()
            row['graph']=json.loads(graph.stats_json());check_graph(row['graph'],tokens);graph.close();graph=None;save()
        report['status']='actual_FA3_projection_witness_collected';save()
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if graph is not None:graph.close()
        if io is not None:io.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
