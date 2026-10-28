"""CPU-only transport of the exact already closed full-chain prefix archive."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';JOB='o128-model-admission-attempt1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    from exact_zip_payload_delta import digest
    names=['exact_zip_payload_delta.py','test_exact_zip_payload_delta.py','run_o128_model_exact_delta.py'];pins={str((M/n).relative_to(ROOT).as_posix()):sha(M/n) for n in names}
    ledger.source_identity=lambda:{'scope':'Explicit owned exact-byte ZIP transport only; no model/native/GPU work and no unrelated untracked reads','owned_source_sha256':pins}
    os.chdir(ROOT/'Garnet');out=ROOT/'results/o128-model-admission-transport';assert not out.exists();out.mkdir();proof=ROOT/'results'/(JOB+'-verified.json');q=json.loads(proof.read_bytes());w=json.loads((ROOT/'results'/JOB/'workflow.json').read_bytes());assert q['exit_code']==w['exit_code'] and q['exit_code'] in (0,1) and w['cleanup_finished']
    source=ROOT/'results/q128-model-admission-attempt1-evidence.zip';target=ROOT/'results'/(JOB+'-evidence.zip');delta=ROOT/'results'/(JOB+'-exact-delta.zip');assert not delta.exists()
    artifacts=[M/n for n in names]+[proof,source,target]
    assert ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_exact_zip_payload_delta'],out/'cpu-ledger','correctness','Exact ZIP headers/order/comment preservation, wrong-parent and corrupted/gapped segment rejection. No GPU.',artifacts)==0
    assert ledger.run([sys.executable,str(M/'exact_zip_payload_delta.py'),'build','--source',str(source),'--target',str(target),'--delta',str(delta),'--expected-target-sha',q['archive_sha256']],out/'build-ledger','correctness','Lossless raw compressed-payload reuse with exact byte comparison. Entire original archive SHA must match; scientific artifacts unchanged.',artifacts)==0
    receipt=dict(status='exact_full_chunk_stage_archive_transport_delta_prepared',delta_sha256=digest(delta),delta_bytes=delta.stat().st_size,source_archive_sha256=digest(source),target_archive_sha256=digest(target),target_archive_bytes=target.stat().st_size,original_archive_bytes_changed=False,GPU_or_model_rerun=False,method_sha256=pins)
    (ROOT/'results'/(JOB+'-exact-delta-receipt.json')).write_bytes(json.dumps(receipt,indent=2).encode());print(json.dumps(receipt))
if __name__=='__main__':main()
