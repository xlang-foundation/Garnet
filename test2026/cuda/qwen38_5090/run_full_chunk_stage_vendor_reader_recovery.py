"""CPU only: preserve the original owner failure and replay its complete saved captures."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    p=json.loads((M/'evidence/full-chunk-stage-vendor-reader-preparation.json').read_bytes());original=json.loads((M/'evidence/full-chunk-stage-vendor-preparation.json').read_bytes());proof=ROOT/'results/full-chunk-stage-vendor-attempt1-verified.json';q=json.loads(proof.read_bytes());w=json.loads((ROOT/'results/full-chunk-stage-vendor-attempt1/workflow.json').read_bytes());assert q['exit_code']==w['exit_code']==1 and w['cleanup_finished'] and sha(proof)==p['original_failure_proof_sha256']
    for n,h in p['files'].items():assert sha(ROOT/n)==h
    for n,h in original['files'].items():assert sha(ROOT/n)==h
    before={n:sha(ROOT/'build/bin'/n) for n in original['runtime']};assert before==original['runtime']
    for n,h in original['checkpoints'].items():assert sha(ROOT/n)==h
    for n,h in original['environment'].items():assert sha(Path(n))==h
    out=ROOT/'results/full-chunk-stage-vendor-reader-recovery';assert not out.exists();out.mkdir();os.chdir(ROOT/'Garnet');ledger.source_identity=lambda:{'scope':'Explicit CPU vendor-reader-only replay, original native source/library/model bytes unchanged; protected unrelated untracked excluded','owned_source_sha256':{n:sha(ROOT/n) for n in p['files']}}
    artifacts=[ROOT/n for n in p['files']]+[proof,M/'evidence/full-chunk-stage-vendor-reader-preparation.json'];code=1;result=dict(status='running',GPU_requested=False,original_GPU_owner_exit_code=1,preparation_sha256=sha(M/'evidence/full-chunk-stage-vendor-reader-preparation.json'))
    try:
        code=ledger.run([sys.executable,'-m','unittest','test2026.cuda.qwen38_5090.test_full_chunk_stage_vendor','test2026.cuda.qwen38_5090.test_full_chunk_stage_vendor_optional_fields'],out/'cpu-ledger','correctness','5CPU exact reader derivation/actual omitted-fields and original finite-byte/readonly/geometry negative controls; no GPU',artifacts)
        if code:raise RuntimeError('Reader controls failed')
        code=ledger.run([sys.executable,str(M/'analyze_full_chunk_stage_vendor_optional_fields.py'),'--proofs',str(ROOT/'results')],out/'replay-ledger','correctness','Reproduce exact original optional-field reader failure, then replay all complete logged primary/readonly/repeats/raw stages and actual printed QG traces; no original GPU outcome relabeled',artifacts)
        if code:raise RuntimeError('Saved-capture reader recovery failed')
        result.update(status='CPU_complete_saved_vendor_captures_recovered',recovery_proof_sha256=sha(ROOT/'results/full-chunk-stage-vendor-optional-fields-verified.json'))
    except BaseException as e:result.update(status='failed',error=repr(e));code=code or 1
    finally:
        result.update(exit_code=code,original_baseline_source_unchanged=all(sha(ROOT/n)==h for n,h in original['files'].items()),runtime_before=before,runtime_after={n:sha(ROOT/'build/bin'/n) for n in original['runtime']},checkpoint_unchanged=all(sha(ROOT/n)==h for n,h in original['checkpoints'].items()),SDK_unchanged=all(sha(Path(n))==h for n,h in original['environment'].items()))
        assert result['original_baseline_source_unchanged'] and result['runtime_before']==result['runtime_after'] and result['checkpoint_unchanged'] and result['SDK_unchanged'];(out/'receipt.json').write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))
    return code
if __name__=='__main__':raise SystemExit(main())
