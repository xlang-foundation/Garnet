"""Snapshot authoritative idle remote baseline before installing combined native source."""
from pathlib import Path
import hashlib,io,json,sys,zipfile
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path(__file__).resolve().parent==M.resolve();sys.path.insert(0,str(M));from run_bf16_low_batch_native_build import idle
    idle();proposal=ROOT/'results/fp8-z128-native-proposal.json';p=json.loads(proposal.read_bytes())
    assert p['status']=='fp8_z128_native_build_proposed_unrun' and len(p['candidate_files'])==28 and len(p['candidate_base'])==28 and p['expected_cpu_test_count']==77
    for name,h in {**p['base_files'],**p['methods']}.items():assert sha(ROOT/name)==h
    for name,h in p['runtime'].items():assert sha(ROOT/'build/bin'/name)==h
    for name,h in p['environment'].items():assert sha(Path(name))==h
    for name,h in p['candidate_base'].items():
        path=ROOT/'Garnet'/name
        if h is None:assert not path.exists()
        else:assert sha(path)==h==p['base_files']['Garnet/'+name]
    assert sum(value is not None for value in p['candidate_base'].values())==8
    bundle=ROOT/p['bundle'];assert sha(bundle)==p['bundle_sha256']
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==28 and set(z.namelist())==set(p['candidate_files'])
        for name,h in p['candidate_files'].items():assert name.startswith('src/') and '..' not in Path(name).parts and hashlib.sha256(z.read(name)).hexdigest()==h
    for job,row in p['parents'].items():
        proof=ROOT/'results'/(job+'-verified.json');q=json.loads(proof.read_bytes());w=json.loads((ROOT/'results'/job/'workflow.json').read_bytes())
        assert sha(proof)==row['proof_sha256'] and q['status']==row['status'] and q['archive_sha256']==row['archive_sha256']==sha(ROOT/'results'/(job+'-evidence.zip')) and w['exit_code']==row['exit_code'] and row['exit_code'] in (0,1) and w['cleanup_finished']
    cache=ROOT/'build/CMakeCache.txt';raw=cache.read_bytes()
    assert b'GARNET_STAGE_VENDOR_DLLS:BOOL=OFF' in raw and b'GARNET_QWEN35_WHOLE_PROGRAM_CUDA:BOOL=OFF' in raw
    objects={path.relative_to(ROOT).as_posix():sha(path) for path in (ROOT/'build/garnet/CMakeFiles/garnet.dir').rglob('*.cu.obj')};assert len(objects)==34
    p.update(status='fp8_z128_native_build_prepared_unrun',cache_before_sha256=sha(cache),cuda_objects_before=objects,proposal=proposal.relative_to(ROOT).as_posix(),proposal_sha256=sha(proposal))
    target=M/'evidence/fp8-z128-native-build-preparation.json';assert not target.exists();target.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'sha256':sha(target),'cuda_objects':34,'existing_targets':8,'absent_targets':20,'source_pins':len(p['base_files'])}))

if __name__=='__main__':main()
