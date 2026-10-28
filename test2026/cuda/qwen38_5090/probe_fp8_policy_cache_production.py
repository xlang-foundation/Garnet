"""Fresh embedded process: production compiled_xmodel cache plus exact tiny GPU output."""
from pathlib import Path
import hashlib,json,os,sys
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    root,out,cache=map(Path,sys.argv[1:4]);index=int(sys.argv[4]);m=root/'Garnet/test2026/cuda/qwen38_5090';sys.path[:0]=[os.environ['GARNET_RUNTIME_ROOT'],str(root/'Garnet')]
    from test2026.cuda.qwen38_5090.fp8_policy_cache_production_contract import CASES,replay_case
    spec=CASES[index];p=json.loads((m/'evidence/fp8-policy-cache-production-preparation.json').read_bytes());assert not out.exists();assert cache.resolve().is_relative_to((root/'results').resolve())
    for n,h in p['installed_files'].items():assert sha(root/n)==h
    library=sha(root/'build/bin/garnet.dll');assert library==p['candidate_library_sha256'];environment={n:os.environ.get(n) for n in spec['environment']};assert environment==spec['environment']
    out.mkdir()
    def inventory(label):
        result={}
        if cache.exists():
            for f in cache.rglob('*'):
                if f.is_file():
                    raw=f.read_bytes();entry=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());snapshot=out/label/f.relative_to(cache);assert snapshot.resolve().is_relative_to(out.resolve());snapshot.parent.mkdir(parents=True,exist_ok=True);assert not snapshot.exists() or snapshot.read_bytes()==raw;snapshot.write_bytes(raw)
                    if f.name=='runtime_graph.cache':entry['fingerprint']=f.read_bytes().splitlines()[1].decode('ascii')
                    result[f.relative_to(cache).as_posix()]=entry
        return result
    row=dict(status='constructing',spec=spec,actual_environment=environment,library_sha256=library,actual_Garnet_load_model_called=False,cache_before=inventory('cache-before'),cache_after=None,runtime_status=None,output_values=None,forward_status=None)
    def save():(out/'report.json').write_bytes(json.dumps(row,indent=2).encode())
    save()
    try:
        import garnet
        row['actual_Garnet_load_model_called']=True;save()
        model=garnet.load_model(str(m/'fp8_policy_cache_fixture.x'),runtime_mode='compiled_xmodel',entry_function='Model',input_shapes=[[1,4]],cache_dir=str(cache))
        assert model is not None;row['runtime_status']=model.runtime_status();save()
        if spec['phase']!='reject':
            value=garnet.tensor_from_host([1.0,2.0,3.0,4.0],dtype='float32',shape=[1,4]);result=model.forward({'inputs':[value]});row['forward_status']=result['status'];assert result['status']=='ok';row['output_values']=garnet.tensor_to_cpu(result['output']).tolist()
        row.update(status='actual_production_cache_case_collected',cache_after=inventory('cache-after'));save();replay_case(row,spec,library);print('ACTUAL_PRODUCTION_CACHE_CASE_PASS '+spec['key'],flush=True)
    except BaseException as e:row.update(status='failed',error=repr(e),cache_after=inventory('cache-after'));save();raise
    return 0
if __name__=='__main__':raise SystemExit(main())
