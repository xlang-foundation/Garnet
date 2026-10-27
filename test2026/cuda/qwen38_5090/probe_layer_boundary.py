"""Ordinary real3968-token T32/T128 controls; read canonical layer3 KV without math changes."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    root,out=map(Path,sys.argv[1:3]);tokens=int(sys.argv[3]);assert tokens in (32,128) and not out.exists()
    m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/layer-boundary-preparation.json').read_bytes())
    for name,h in p['installed_files'].items():assert sha((root/name).read_bytes())==h
    assert sha((root/'build/bin/garnet.dll').read_bytes())==p['candidate_library_sha256']
    for flag in ('GARNET_FP8_GDN_M32_CANONICAL','GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_PARALLEL_BATCH_TOP1'):assert os.environ.get(flag)=='1'
    for flag in ('GARNET_FP8_GDN_M128_CANONICAL','GARNET_FP8_GDN_Z128_CANONICAL'):assert os.environ.get(flag)==('1' if tokens==128 else '0')
    for flag in ('GARNET_FP8_GDN_Z128_RECONSTRUCT','GARNET_FP8_FULL_ATTN_KV128_CANONICAL'):assert os.environ.get(flag)==('1' if tokens==128 else '0')
    assert os.environ.get('GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES')=='0' and os.environ.get('GARNET_NVFP4_DOWN_CANDIDATE')=='0'
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')]
    import garnet
    from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder,create_cuda_bf16_prefill128_study_decoder
    from test2026.cuda.qwen38_5090.kv128_prefix_cache_witness import PREFIXES,parts,cache_components,same_state,check_graphs
    from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
    parent='kv128-prefix-cache-attempt1';phase='baseline' if tokens==32 else 'native'
    with zipfile.ZipFile(root/'results'/(parent+'-evidence.zip')) as z:
        base='results/'+parent+'/'+phase+'/';old=json.loads(z.read(base+'report.json'));expected=old;oldraw={name.removeprefix(base):z.read(name) for name in z.namelist() if name.startswith(base) and name.endswith('.f32')}
    case=json.loads((m/'evidence/native-host-stage-cases.json').read_bytes())['cases'][0];ids=case['prompt_ids'];assert case['case']=='text' and len(ids)==3968 and ids==expected['prompt_ids']
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes());out.mkdir();decoder=None
    report={'boundaries':[],'admission_oracle_job':'mtp-prefill16-full4k-admission-attempt1','same_profile_primary_equivalence_passed':False,'status':'constructing','tokens':tokens,'prompt_ids':ids,'reference_job':parent,'candidate_library_sha256':p['candidate_library_sha256'],'snapshots':[],'performance_admitted':False,'cause_established':False,'scope':'Read-only hidden boundaries after layers0..4 of the actual full3968 chain, two exact repeats, and complete layer3 KV at six prefixes. Preserve primary same-profile state/vector/cache equivalence to the retained current run; T32 also retains128IDs/finalstate. Failed T128 is instrumentation-equivalence control only, never admission oracle. Original correct T32 admission target unchanged. No native/production changes or timing admission.'}
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        factory=create_cuda_bf16_low_batch_study_decoder if tokens==32 else create_cuda_bf16_prefill128_study_decoder
        from test2026.cuda.qwen38_5090.layer_boundary_program import construct_with_boundary_program
        decoder=construct_with_boundary_program(factory,str(checkpoint),config);decoder.enable_cuda_graph(True);io=decoder.io;layout=decoder.programs[4].layout;width=decoder.programs[4].state_width;offsets=cache_components(layout)
        report['graphs']={str(t):json.loads(program.graph.stats_json()) for t,program in decoder.programs.items()};check_graphs(report['graphs'],tokens)
        def capture(value,offset,words,label):
            rows=[];digest=hashlib.sha256()
            for start,count in parts(words):
                part=io.zeros([1,count]);io.slice_into(value,offset+start,part);raw=bytes(io.finite_f32_bytes(part));assert len(raw)==count*4
                name=label+'-'+str(start)+'.f32';target=out/name;assert not target.exists();target.write_bytes(raw);digest.update(raw);rows.append({'file':name,'offset':start,'words':count,'sha256':sha(raw)})
            return {'words':words,'sha256':digest.hexdigest(),'parts':rows}
        def state(label):
            assert decoder.state_transaction is None and not decoder.poisoned
            logical=logical_state_witness(io,decoder.state,layout,1,width,decoder.slots,packed_bf16=True);vectors={}
            for name,value in (('hidden',decoder.last_hidden),('normalized',decoder.head.run_device([decoder.last_hidden]))):
                raw=bytes(io.finite_f32_bytes(value));assert len(raw)==20480;filename=label+'-'+name+'.f32';assert not (out/filename).exists();(out/filename).write_bytes(raw);vectors[name]={'file':filename,'bytes':len(raw),'sha256':sha(raw)}
            return {'logical':logical,'vectors':vectors}
        read=lambda name:(out/name).read_bytes()
        decoder.reset();report['empty_state']=state('empty');save();same_state(report['empty_state'],old['empty_state'],read,oldraw.__getitem__)
        embeddings=io.embedding(ids);hidden=[io.row(embeddings,i) for i in range(3968)];positions=[[[i]*3 for i in range(3968)]]
        decoder.prefill_mode=True
        try:
            for start in range(0,3968,tokens):
                end=start+tokens;prediction=decoder.chunk_hidden(hidden[start:end],[[1] for _ in range(tokens)],[positions[0][start:end]],end==3968)
                assert decoder.slots==[end]
                program=decoder.programs[tokens];boundary={'start':start,'count':tokens,'layers':{}}
                for layer,value in zip(program.boundary_layers,program.boundary_buffers):
                    raw=bytes(io.finite_f32_bytes(value));assert len(raw)==tokens*5120*4;filename='boundary-'+str(start)+'-L'+str(layer)+'.f32';assert not (out/filename).exists();(out/filename).write_bytes(raw)
                    boundary['layers'][str(layer)]={'file':filename,'words':tokens*5120,'sha256':sha(raw)}
                report['boundaries'].append(boundary)
                if end in PREFIXES:
                    snapshot={'prefix':end,'slots':list(decoder.slots),'caches':{name:capture(decoder.state,offset,2097152,'p'+str(end)+'-'+name) for name,offset in offsets.items()}}
                    report['snapshots'].append(snapshot);save();print('PREFIX_CAPTURED',tokens,end,flush=True)
        finally:decoder.prefill_mode=False
        report['ordinary_prefill']=state('ordinary-prefill');save();same_state(report['ordinary_prefill'],expected['ordinary_prefill'],read,oldraw.__getitem__)
        # Same-profile parity is diagnostic only. Correct T32 remains admission oracle.
        from test2026.cuda.qwen38_5090.kv128_prefix_cache_witness import snapshots
        assert snapshots(report,read)==snapshots(expected,oldraw.__getitem__)
        ordinary=[];slot=3968
        for step in range(128 if tokens==32 else 0):
            ordinary.append(prediction[0])
            if step+1<128:prediction=decoder.step(prediction,[1],[[slot]*3]);slot+=1
        if tokens==32:
            report['ordinary_ids']=ordinary;report['ordinary_final']=state('ordinary-final');save()
            assert ordinary==expected['ordinary_ids'] and decoder.slots==[4095];same_state(report['ordinary_final'],expected['ordinary_final'],read,oldraw.__getitem__)
        report['same_profile_primary_equivalence_passed']=True;save()
        decoder.reset();decoder.prefill_mode=True
        try:
            for index,start in enumerate(range(0,3968,tokens)):
                end=start+tokens;decoder.chunk_hidden(hidden[start:end],[[1] for _ in range(tokens)],[positions[0][start:end]],end==3968)
                assert decoder.slots==[end]
                for layer,value in zip(decoder.programs[tokens].boundary_layers,decoder.programs[tokens].boundary_buffers):
                    raw=bytes(io.finite_f32_bytes(value));assert len(raw)==tokens*5120*4
                    value=report['boundaries'][index]['layers'][str(layer)];value['repeat_sha256']=sha(raw);assert value['repeat_sha256']==value['sha256']
                if end in PREFIXES:print('BOUNDARY_REPEAT',tokens,end,flush=True);save()
        finally:decoder.prefill_mode=False
        report['repeat_prefill']=state('repeat-prefill');same_state(report['repeat_prefill'],report['ordinary_prefill'],read,read)
        report['repeat_complete_prefill_state_vectors_exact']=True;report['boundary_runs']=decoder.programs[tokens].boundary_runs
        assert [row['prefix'] for row in report['snapshots']]==list(PREFIXES)
        report.update(status='layer_boundaries_collected_primary_profile_exact',instrumentation_equivalence_only=True,model_admission_passed=(tokens==32));save();return 0
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if decoder is not None:decoder.close()
if __name__=='__main__':raise SystemExit(main())
