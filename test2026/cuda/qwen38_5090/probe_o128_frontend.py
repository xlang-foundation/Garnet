"""One fresh embedded process calling actual Garnet decoder-layer capture/lowering."""
from pathlib import Path
import hashlib,json,os,sys
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root,out,index=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]);m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')]
    from test2026.cuda.qwen38_5090.o128_frontend_contract import CASES,replay_case
    spec=CASES[index];p=json.loads((m/'evidence/o128-frontend-preparation.json').read_bytes());assert not out.exists()
    for n,h in p['installed_files'].items():assert sha(root/n)==h
    library=sha(root/'build/bin/garnet.dll');assert library==p['candidate_library_sha256']
    for n,h in p['runtime'].items():assert sha(root/'build/bin'/n)==(library if n=='garnet.dll' else h)
    for n,h in p['checkpoints'].items():assert sha(root/n)==h
    observed={n:os.environ.get(n) for n in spec['flags']};assert observed==spec['flags']
    import garnet,tensor
    from xModel.qwen35.vl_0_8b.cuda_chunk import Qwen35CUDAChunk,T
    cfg=json.loads((root/'models/Qwen3.8-27B-NVFP4/config.json').read_bytes())['text_config'];assert cfg['hidden_size']==5120 and cfg['num_attention_heads']==24 and cfg['num_key_value_heads']==4 and cfg['head_dim']==256
    b,t=spec['batch'],spec['tokens'];shapes=([b,t,spec['hidden']],[b,4,4096,256],[b,4,4096,256],[b,t],[b,t,3],[b,t]);symbols=[tensor.input('frontend_'+str(i),shape=s,dtype=tensor.float32) for i,s in enumerate(shapes)]
    @T.fusion(name='qwen38_real_fp8_frontend_contract',boundary='required')
    def forward(h,k,v,a,pos,slot):return Qwen35CUDAChunk(h,k,v,a,pos,slot,cfg,spec['layer'],'full_attention',fused_attention=True)
    row=dict(spec=spec,actual_flags=observed,library_sha256=library,actual_Garnet_frontend_requested=True,status='constructing',compiled=False,graph_stats=None,native_error=None);out.mkdir()
    def save():(out/'report.json').write_bytes(json.dumps(row,indent=2).encode())
    graph=None;save()
    try:
        graph=garnet.compiled_graph();graph.set_builder_workspace_bytes(128<<20)
        try:graph.build(forward(*symbols),symbols,str(root/'models/Qwen3.8-27B-NVFP4'))
        except Exception as e:
            row.update(status='native_build_rejected',native_error=str(e));save()
            if spec['expected_error'] is None:raise
            assert spec['expected_error'] in str(e),('Unexpected native rejection',spec['key'],str(e))
            row['status']='expected_native_rejection'
        else:
            row.update(status='actual_frontend_compiled',compiled=True,graph_stats=json.loads(graph.stats_json()));save()
            assert spec['expected_error'] is None,('Native frontend accepted invalid case',spec['key'])
        replay_case(row,spec);save();print('ACTUAL_O128_FRONTEND_CASE_PASS '+spec['key'],flush=True)
    except BaseException as e:row.update(status='failed',error=repr(e));save();raise
    finally:
        if graph is not None:graph.close()
    return 0
if __name__=='__main__':raise SystemExit(main())
