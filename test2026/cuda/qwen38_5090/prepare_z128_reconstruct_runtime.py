from pathlib import Path
import hashlib,json,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda raw:hashlib.sha256(raw).hexdigest()
def main():
    sys.path.insert(0,str(M));from run_bf16_low_batch_native_build import idle
    proposal=ROOT/'results/z128-reconstruct-runtime-proposal.json';p=json.loads(proposal.read_bytes());assert p['status']=='separate_reconstruction_runtime_origin_proposed_unrun';idle()
    for name,h in {**p['files'],**p['methods']}.items():assert sha((ROOT/name).read_bytes())==h
    for name,h in p['runtime'].items():assert sha((ROOT/'build/bin'/name).read_bytes())==h
    for name,h in p['environment'].items():assert sha(Path(name).read_bytes())==h
    parent=ROOT/'results/z128-reconstruct-construction-attempt2';assert sha(parent.with_name(parent.name+'-quarantine-verified.json').read_bytes())==p['parent_proof_sha256'] and sha(parent.with_name(parent.name+'-evidence.zip').read_bytes())==p['parent_archive_sha256']
    p.update(status='separate_reconstruction_runtime_origin_prepared_unrun',proposal_sha256=sha(proposal.read_bytes()));target=M/'evidence/z128-reconstruct-runtime-preparation.json';assert not target.exists();target.write_bytes(json.dumps(p,indent=2).encode());print(json.dumps({'status':p['status'],'preparation_sha256':sha(target.read_bytes()),'source_pins':len(p['files']),'method_pins':len(p['methods'])}))
if __name__=='__main__':main()
