"""Ordinary real3968-token T32/T128 controls; read canonical layer3 KV without math changes."""
from pathlib import Path
import hashlib,json,os,sys,zipfile
sha=lambda raw:hashlib.sha256(raw).hexdigest()

def main():
    root,out=map(Path,sys.argv[1:3]);tokens=int(sys.argv[3]);assert tokens in (32,128) and not out.exists()
    m=root/'Garnet/test2026/cuda/qwen38_5090';p=json.loads((m/'evidence/fa3-prefix-cache-witness-preparation.json').read_bytes())
    for name,h in p['installed_files'].items():assert sha((root/name).read_bytes())==h
    assert sha((root/'build/bin/garnet.dll').read_bytes())==p['candidate_library_sha256']
    for flag in ('GARNET_FP8_GDN_M32_CANONICAL','GARNET_FP8_FULL_ATTN_M32_CANONICAL','GARNET_PARALLEL_BATCH_TOP1'):assert os.environ.get(flag)=='1'
    for flag in ('GARNET_FP8_GDN_M128_CANONICAL','GARNET_FP8_GDN_Z128_CANONICAL'):assert os.environ.get(flag)==('1' if tokens==128 else '0')
    assert os.environ.get('GARNET_QWEN38_BF16_LOW_BATCH_TILED_SCORES')=='0' and os.environ.get('GARNET_NVFP4_DOWN_CANDIDATE')=='0'
    sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')]
    import garnet
    from xModel.qwen38.vl_27b.model import create_cuda_bf16_low_batch_study_decoder,create_cuda_bf16_prefill128_study_decoder
    from test2026.cuda.qwen38_5090.fa3_prefix_cache_witness import PREFIXES,parts,cache_components,same_state,check_graphs
    from test2026.cuda.qwen38_5090.bf16_model_state_witness import logical_state_witness
    parent='mtp-prefill16-full4k-admission-attempt1' if tokens==32 else 'fp8-z128-full4k-model-admission-attempt1'
    with zipfile.ZipFile(root/'results'/(parent+'-evidence.zip')) as z:
        base='results/'+parent+'/native/';old=json.loads(z.read(base+'report.json'));expected=old['cases'][0] if tokens==32 else old['ordinary_observations'][0];oldraw={name.removeprefix(base):z.read(name) for name in z.namelist() if name.startswith(base) and name.endswith('.f32')}
    case=json.loads((m/'evidence/native-host-stage-cases.json').read_bytes())['cases'][0];ids=case['prompt_ids'];assert case['case']=='text' and len(ids)==3968 and ids==expected['prompt_ids']
    checkpoint=root/'models/Qwen3.8-27B-NVFP4';config=json.loads((checkpoint/'config.json').read_bytes());out.mkdir();decoder=None
    report={'status':'constructing','tokens':tokens,'prompt_ids':ids,'reference_job':parent,'candidate_library_sha256':p['candidate_library_sha256'],'snapshots':[],'performance_admitted':False,'cause_established':False,'scope':'Read-only canonical layer3 KV captures at six prefixes during same ordinary3968+128 text case. Retained T32 baseline and prior failed T128 final states/IDs must reproduce exactly. No mathematical/model/production source change; no quality or timing admission.'}
    def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
    save()
    try:
        factory=create_cuda_bf16_low_batch_study_decoder if tokens==32 else create_cuda_bf16_prefill128_study_decoder
        decoder=factory(str(checkpoint),config);decoder.enable_cuda_graph(True);io=decoder.io;layout=decoder.programs[4].layout;width=decoder.programs[4].state_width;offsets=cache_components(layout)
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
                if end in PREFIXES:
                    snapshot={'prefix':end,'slots':list(decoder.slots),'caches':{name:capture(decoder.state,offset,2097152,'p'+str(end)+'-'+name) for name,offset in offsets.items()}}
                    report['snapshots'].append(snapshot);save();print('PREFIX_CAPTURED',tokens,end,flush=True)
        finally:decoder.prefill_mode=False
        report['ordinary_prefill']=state('ordinary-prefill');save();same_state(report['ordinary_prefill'],expected['ordinary_prefill'],read,oldraw.__getitem__)
        ordinary=[];slot=3968
        for step in range(128):
            ordinary.append(prediction[0])
            if step+1<128:prediction=decoder.step(prediction,[1],[[slot]*3]);slot+=1
        report['ordinary_ids']=ordinary;report['ordinary_final']=state('ordinary-final');save()
        assert ordinary==expected['ordinary_ids'] and decoder.slots==[4095];same_state(report['ordinary_final'],expected['ordinary_final'],read,oldraw.__getitem__)
        assert [row['prefix'] for row in report['snapshots']]==list(PREFIXES)
        report.update(status='fa3_prefix_caches_collected_retained_ordinary_control_exact',retained_empty_prefill_final_vectors_and_states_exact=True,retained_128_output_ids_exact=True);save();return 0
    except BaseException as error:report.update(status='failed',error=repr(error));save();raise
    finally:
        if decoder is not None:decoder.close()
if __name__=='__main__':raise SystemExit(main())
