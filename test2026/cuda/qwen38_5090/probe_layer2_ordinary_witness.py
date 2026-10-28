"""Actual real-input stage witness; only same-profile primary parity permits attribution."""
from pathlib import Path
import hashlib,json,os,struct,sys
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    root,out=map(Path,sys.argv[1:3]);m=root/'Garnet/test2026/cuda/qwen38_5090';repo=root/'Garnet';assert not out.exists()
    p=json.loads((m/'evidence/layer2-ordinary-witness-preparation.json').read_bytes());assert sha((root/'build/bin/garnet.dll').read_bytes())==p['candidate_library_sha256']
    for name,h in p['installed_files'].items():assert sha((root/name).read_bytes())==h
    for name,h in p['checkpoints'].items():assert sha((root/name).read_bytes())==h
    for name,h in p['runtime'].items():
        if name!='garnet.dll':assert sha((root/'build/bin'/name).read_bytes())==h
    for name in ('GARNET_FP8_GDN_M32_CANONICAL','GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_FP8_GDN_M128_CANONICAL'):assert os.environ.get(name)=='1'
    assert os.environ.get('GARNET_NVFP4_DOWN_CANDIDATE')=='0'
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(repo)]
    import garnet,tensor
    from xModel.qwen35.vl_0_8b.decoder_layer import T
    from test2026.cuda.qwen38_5090.layer2_ordinary_layout import layout,pieces,check_graph,validate_profile,INPUT_SOURCE
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';case_raw=(m/'evidence/native-host-stage-cases.json').read_bytes();case=json.loads(case_raw)['cases'][0]
    assert case['case']=='text' and len(case['prompt_ids'])==3968;ids=case['prompt_ids'][:128];assert len(set(ids))>1
    out.mkdir();report={'status':'constructing','candidate_library_sha256':p['candidate_library_sha256'],'cases_sha256':sha(case_raw),'config_sha256':sha((checkpoint/'config.json').read_bytes()),'prompt_tokens':3968,'input_ids':ids,'input_source':INPUT_SOURCE,'input_fixture_sha256':p['common_input_sha256'],'frontend_rejections':[],'cases':[],'model_or_performance_qualified':False,'scope':p['scope']}
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    io=graph=None;pool={};weights={};save()
    def capture(value,offset,words,label):
        rows=[];digest=hashlib.sha256()
        for start,count in pieces(words):
            part=io.zeros([1,count]);io.slice_into(value,offset+start,part);raw=bytes(io.finite_f32_bytes(part));assert len(raw)==count*4
            name=label+'-part-'+str(start)+'.f32';target=out/name;assert not target.exists();target.write_bytes(raw);digest.update(raw)
            rows.append({'file':name,'offset':start,'words':count,'sha256':sha(raw)})
        return {'words':words,'sha256':digest.hexdigest(),'parts':rows}
    def raw(entry):
        values=[]
        for piece in entry['parts']:
            values.append((out/piece['file']).read_bytes())
        return b''.join(values)
    def layer_weights(layer):
        if layer in weights:return weights[layer]
        prefix='model.language_model.layers.'+str(layer);rows=[]
        for suffix in ('.mlp.gate_up_proj.weight','.mlp.down_proj.weight','.linear_attn.in_proj_qkv.weight','.linear_attn.in_proj_z.weight','.linear_attn.out_proj.weight'):
            name=prefix+suffix;info=json.loads(io.compressed_weight_info(name));buffers=[('packed',info['packed_shape'],io.compressed_weight(name))]
            if suffix=='.mlp.gate_up_proj.weight':assert info['joined_mlp'] and info['source_weights']==[prefix+'.mlp.gate_proj.weight',prefix+'.mlp.up_proj.weight']
            if info['scale_bytes']:buffers.append(('scales',info['scale_shape'],io.compressed_scales(name)))
            assert len(buffers)==(2 if suffix.startswith('.mlp.') else 1);rows.append((name,info,buffers))
        report.setdefault('weight_metadata',{})[str(layer)]=[{k:v for k,v in info.items()} for name,info,buffers in rows];weights[layer]=rows;save();return rows
    def build(layer,tokens,diagnostic,bad=None):
        validate_profile(layer,tokens,diagnostic);assert bad is None
        bad=bad or {};batch=bad.get('batch',1);source_tokens=bad.get('tokens',tokens);kind=tensor.bfloat16 if bad.get('bf16') else tensor.float32
        shapes=([batch,source_tokens,5120],[1,10240,4],[1,48,128,128],[1,source_tokens]);symbols=[tensor.input('gdn_input_'+str(i),shape=shape,dtype=kind if i==0 else tensor.float32) for i,shape in enumerate(shapes)]
        rows=layer_weights(layer);values=[];slots=[]
        for name,info,buffers in rows:
            current=[]
            for role,shape,value in buffers:
                current.append(len(values));values.append(value);symbols.append(tensor.input('gdn_weight_'+str(len(values)),shape=shape,dtype=tensor.int32))
            slots.append((name,current))
        @T.fusion(name='qwen38_gdn_chunk_stage_witness',boundary='required')
        def forward(hidden,history,state,active,*inputs):
            compressed={name:[inputs[i] for i in indices] for name,indices in slots}
            bound=hidden*T.binary_op('qwen35_conv_bind_state')*history
            bound=bound*T.binary_op('qwen35_delta_bind_state')*state
            if not bad.get('missing_active'):bound=bound*T.binary_op('qwen35_conv_bind_active')*active
            return bound*T.unary_op('qwen35_linear_decoder_chunk',layer_prefix='model.language_model.layers.'+str(bad.get('layer',layer)),key_heads=16,value_heads=48,key_dim=128,value_dim=128,eps=1.e-6,fused_delta=1,register_delta_128=1,retain_linear_inputs=0,joined_input_projections=0,joined_mlp_projections=1,external_compressed_weights=compressed,diagnostic_linear_stages=bad.get('legacy',0),diagnostic_gdn_chunk_stages=bad.get('flag',int(diagnostic)))
        result=garnet.compiled_graph();result.set_builder_workspace_bytes(256<<20)
        try:result.build(forward(*symbols),symbols,str(checkpoint));result.enable_cuda_graph(True)
        except BaseException:result.close();raise
        return result,values
    try:
        io=garnet.cuda_tensor_io();io.open(str(checkpoint));io.enable_nvfp4_head(True);io.enable_host_embedding(True)
        assert os.environ.get('GARNET_FP8_GDN_Z128_CANONICAL')=='1'
        raw_input=(m/'evidence/layer2-ordinary-common-input.f32').read_bytes();assert len(raw_input)==2621440 and sha(raw_input)==p['common_input_sha256'];prompt=io.from_host(list(struct.unpack('<655360f',raw_input)),[1,128*5120]);report['common_input']=capture(prompt,0,128*5120,'common-input');save();assert raw(report['common_input'])==raw_input
        layer_input=prompt
        for layer in (2,):
            retained=None
            for tokens in (32,128):
                ordinary=None
                for diagnostic in (False,):
                    label='l'+str(layer)+'-t'+str(tokens)+'-d'+str(int(diagnostic));graph,params=build(layer,tokens,diagnostic);spec=layout(tokens,diagnostic)
                    hidden=io.zeros([1,tokens,5120]);history=io.zeros([1,10240,4]);state=io.zeros([1,48,128,128]);active=io.from_host([1.0]*tokens,[1,tokens]);args=[hidden,history,state,active]+params
                    row={'layer':layer,'tokens':tokens,'diagnostic':diagnostic,'layout':spec,'input_lineage':INPUT_SOURCE,'active':capture(io.storage_view(active,[1,tokens]),0,tokens,label+'-active'),'repeats':[]};report['cases'].append(row);save()
                    flat_hidden=io.storage_view(hidden,[1,tokens*5120]);flat_history=io.storage_view(history,[1,40960]);flat_state=io.storage_view(state,[1,786432])
                    for warmup in range(3):
                        io.zero(history);io.zero(state);io.slice_into(layer_input,0,flat_hidden);result=graph.run_device(args);boundary=io.zeros([1,1]);io.slice_into(result,0,boundary);assert len(bytes(io.finite_f32_bytes(boundary)))==4
                    row['graph']=json.loads(graph.stats_json());check_graph(row['graph'],tokens,diagnostic);save()
                    prior_repeat=None;output128=io.zeros([1,128*5120]) if layer==0 and tokens==32 and not diagnostic else None
                    for repeat in range(2):
                        io.zero(history);io.zero(state);sequence=[];row['repeats'].append(sequence)
                        for start in range(0,128,tokens):
                            stem=label+'-r'+str(repeat)+'-s'+str(start);io.slice_into(layer_input,start*5120,flat_hidden)
                            chunk={'start':start,'repeat':repeat,'input_hidden':capture(flat_hidden,0,tokens*5120,stem+'-in-hidden'),'input_history':capture(flat_history,0,40960,stem+'-in-history'),'input_state':capture(flat_state,0,786432,stem+'-in-state'),'arrays':{}};sequence.append(chunk);save()
                            result=graph.run_device(args)
                            for part in spec['rows']:chunk['arrays'][part['name']]=capture(result,part['offset'],part['words'],stem+'-'+part['name'])
                            chunk['input_hidden_after']=capture(flat_hidden,0,tokens*5120,stem+'-after-hidden');chunk['input_history_after']=capture(flat_history,0,40960,stem+'-after-history');chunk['input_state_after']=capture(flat_state,0,786432,stem+'-after-state');save()
                            assert raw(chunk['input_hidden'])==raw(chunk['input_hidden_after']) and raw(chunk['input_history'])==raw(chunk['input_history_after']) and raw(chunk['input_state'])==raw(chunk['input_state_after'])
                            if output128 is not None:
                                # Copy bounded output token rows to stable common layer1 input.
                                token_row=io.zeros([1,5120]);output_view=io.storage_view(output128,[1,128,5120])
                                for token in range(tokens):io.slice_into(result,token*5120,token_row);io.write_chunk_token(token_row,output_view,start+token)
                            io.slice_into(result,tokens*5120,flat_history);io.slice_into(result,tokens*5120+40960,flat_state)
                        actual=[raw(chunk['arrays']['primary']) for chunk in sequence]
                        if prior_repeat is not None:assert actual==prior_repeat
                        prior_repeat=actual
                        if diagnostic:
                            row['primary_matches_ordinary']=actual==ordinary;save();assert row['primary_matches_ordinary'],('Diagnostic changes primary',layer,tokens)
                    row['graph']=json.loads(graph.stats_json());check_graph(row['graph'],tokens,diagnostic);save()
                    if not diagnostic:ordinary=prior_repeat
                    if output128 is not None:retained=output128
                    graph.close();graph=None
            if layer==0:assert retained is not None;layer_input=retained
        report['status']='layer2_ordinary_collected_needs_replay';save()
    except BaseException as error:
        report.update(status='failed',error=repr(error));save()
        if 'buildSerializedNetwork failed' in report['error']:
            try:
                from test2026.cuda.qwen38_5090.fp8_z128_native_audit_bridge import observe
                target=root/'results/fp8-z128-native-package-audit-attempt1/layer2-ordinary-attempt1-context.json'
                report['initialization_context_audit']=observe(root,target)
                copy=out/'context-audit.json';assert not copy.exists();copy.write_bytes(target.read_bytes());save()
            except BaseException as audit_error:report['initialization_audit_error']=repr(audit_error);save()
        raise
    finally:
        if graph is not None:graph.close()
        if io is not None:io.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
