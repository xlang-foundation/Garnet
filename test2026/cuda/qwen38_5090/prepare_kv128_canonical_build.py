"""Check the idle authoritative machine before the isolated native candidate build."""
from pathlib import Path
import hashlib,json,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path.insert(0,str(M));from run_bf16_low_batch_native_build import idle
    idle();proposal=ROOT/'results/kv128-canonical-proposal.json';p=json.loads(proposal.read_bytes())
    assert p['status']=='KV128_canonical_build_proposed_unrun' and len(p['candidate_files'])==50
    for name,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/name)==h,('Source drift',name)
    for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in p['environment'].items():assert sha(Path(name))==h
    for name,h in p['candidate_base'].items():
        path=ROOT/'Garnet'/name
        if h is None:assert not path.exists(),('Unowned target',name)
        else:assert sha(path)==h==p['base_files']['Garnet/'+name]
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes())
        assert sha(proof)==row['proof_sha256'] and q['archive_sha256']==row['archive_sha256']==sha(ROOT/'results'/(job+'-evidence.zip'))
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==50 and set(z.namelist())==set(p['candidate_files'])
        for name,h in p['candidate_files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    cache=ROOT/'build/CMakeCache.txt';assert all(flag in cache.read_bytes() for flag in (b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF',b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF'))
    objects={path.relative_to(ROOT).as_posix():sha(path) for path in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert len(objects)==34
    p.update(status='KV128_canonical_build_prepared_unrun',cache_before_sha256=sha(cache),cuda_objects_before=objects,proposal=proposal.relative_to(ROOT).as_posix(),proposal_sha256=sha(proposal))
    target=M/'evidence/kv128-canonical-native-build-preparation.json';assert not target.exists();target.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'source_pins':len(p['base_files']),'method_pins':len(p['methods']),'candidate_files':50,'existing_targets':sum(v is not None for v in p['candidate_base'].values())}))
if __name__=='__main__':main()
