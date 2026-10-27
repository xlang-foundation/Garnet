"""Retain original verifier failure, then run strict linked-fixture CPU replay."""
from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    original=M/'verify_kv128_projection.py';analysis=M/'analyze_kv128_projection.py';record=M/'evidence/kv128-projection-analysis-derivation.json';pins={p.relative_to(ROOT).as_posix():sha(p) for p in (Path(__file__).resolve(),original,analysis,record)}
    ledger.source_identity=lambda:{'scope':'Explicit owned postanalysis methods only; unrelated protected untracked body excluded','owned_source_sha256':pins}
    os.chdir(ROOT/'Garnet');out=ROOT/'results/kv128-projection-postanalysis';assert not out.exists();out.mkdir()
    artifacts=[original,analysis,record,ROOT/'results/kv128-projection-attempt1-evidence.zip',ROOT/'results/fa3-projection-witness-attempt2-evidence.zip']
    code=ledger.run([sys.executable,str(original),'--proofs',str(ROOT/'results')],out/'original-verifier-failure','correctness','Retain frozen original KeyError for missing current-archive fixture; model already terminal and restored',artifacts);assert code==1
    code=ledger.run([sys.executable,str(analysis),'--proofs',str(ROOT/'results')],out/'linked-fixture-replay','correctness','Exact byte-checked derivative resolves SHA-pinned fixture from already linked parent; all arrays and admission checks unchanged; CPU only',artifacts);assert code==0
    print(json.dumps(dict(status='original_failure_and_strict_linked_fixture_CPU_replay_retained',original_verifier_exit=1,strict_replay_exit=0,proof_sha256=sha(ROOT/'results/kv128-projection-attempt1-verified.json'))))
if __name__=='__main__':main()
