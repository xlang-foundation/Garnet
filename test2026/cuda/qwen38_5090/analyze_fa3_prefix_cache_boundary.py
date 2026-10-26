"""Additional independent source-bundle/original-source and final-cache/state linkage replay."""
from pathlib import Path
import argparse,hashlib,io,json,sys,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.verify_fa3_prefix_cache_witness import JOB,M,checked
from test2026.cuda.qwen38_5090.fa3_prefix_cache_witness import read_cache
from test2026.cuda.qwen38_5090.bf16_float_reference_state import expand_bf16_carrier
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--proofs',type=Path,required=True);a=parser.parse_args();archive=a.proofs/(JOB+'-evidence.zip');z,manifest=checked(archive)
    try:
        base='results/'+JOB+'/';p=json.loads(z.read(M+'evidence/fa3-prefix-cache-witness-preparation.json'));w=json.loads(z.read(base+'workflow.json'));proof=(a.proofs/(JOB+'-verified.json')).read_bytes();q=json.loads(proof)
        assert q['status']=='fa3_prefix_cache_controls_replayed' and q['archive_sha256']==sha(archive.read_bytes()) and q['retained_ordinary_controls_exact'] and w['exit_code']==0
        assert w['source_before_sha256']==w['source_after_sha256']==p['files'] and w['installed_source_sha256']==p['installed_files'] and w['cleanup_finished'] and w['baseline_source_restored'] and w['baseline_library_restored']
        source_raw=z.read(p['source_bundle']);assert sha(source_raw)==p['source_bundle_sha256']
        with zipfile.ZipFile(io.BytesIO(source_raw)) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==35 and set(source.namelist())==set(p['candidate_files'])
            for name,h in p['candidate_files'].items():assert sha(source.read(name))==h
        saved={name:h for name,h in p['candidate_base'].items() if h is not None};assert len(saved)==12
        with zipfile.ZipFile(io.BytesIO(z.read(base+'original-source.zip'))) as source:
            assert source.testzip() is None and len(source.namelist())==len(set(source.namelist()))==12 and set(source.namelist())==set(saved)
            for name,h in saved.items():assert sha(source.read(name))==h==p['files']['Garnet/'+name]
        links=[]
        for phase in ('baseline','native'):
            current=base+phase+'/';r=json.loads(z.read(current+'report.json'));last=r['snapshots'][-1];assert last['prefix']==3968
            for role,name in ((1,'key'),(2,'value')):
                raw=read_cache(last['caches'][name],lambda file:z.read(current+file));logical=sha(expand_bf16_carrier(raw));component=r['ordinary_prefill']['logical']['components'][6+role-1]
                assert component['layer']==3 and component['role']==role and component['logical_words']==4194304 and component['row_sha256']==[logical]
                links.append({'phase':phase,'component':name,'physical_bytes':len(raw),'complete_logical_f32_bytes':len(raw)*2,'complete_logical_sha256':logical,'retained_full_state_link_exact':True})
        result={'status':'fa3_prefix_source_and_final_cache_state_links_replayed','archive_sha256':q['archive_sha256'],'original_proof_sha256':sha(proof),'candidate_sources_verified':35,'original_sources_verified':12,'new_targets_removed':23,'complete_final_cache_state_links':links,'first_sampled_different_prefix':q['first_sampled_different_prefix'],'prefixes':q['prefixes'],'source_runtime_SDK_checkpoint_restored':True,'cause_established':False,'performance_admitted':False,'two_times_target_achieved':False,'analyzer_sha256':sha(Path(__file__).read_bytes())}
    finally:z.close()
    raw=json.dumps(result,indent=2).encode();target=a.proofs/(JOB+'-analysis.json');assert not target.exists() or target.read_bytes()==raw
    if not target.exists():target.write_bytes(raw)
    print(json.dumps({'status':result['status'],'proof_sha256':sha(raw),'first_sampled_different_prefix':q['first_sampled_different_prefix']}));return 0
if __name__=='__main__':raise SystemExit(main())
