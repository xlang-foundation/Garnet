from pathlib import Path
import hashlib,json,os,sys
ROOT=Path('D:/CantorAI/5090-study-20261023');M=ROOT/'Garnet/test2026/cuda/qwen38_5090';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sys.path[:0]=[str(M),str(ROOT/'Garnet')];import run_logged as ledger
    p=json.loads((M/'evidence/q128-canonical-build-preparation.json').read_bytes());w=json.loads((ROOT/'results/q128-canonical-build-attempt1/workflow.json').read_bytes());assert w['exit_code']==0 and w['cleanup_finished'] and w['baseline_library_restored'] and w['baseline_source_restored'];names=['verify_q128_canonical_native_build_parent.py','run_q128_build_parent_reader.py','evidence/q128-build-parent-reader-derivation.json'];pins={**p['methods'],**{'Garnet/test2026/cuda/qwen38_5090/'+n:sha(M/n) for n in names}}
    for n,h in pins.items():assert sha(ROOT/n)==h
    out=ROOT/'results/q128-build-parent-reader';assert not out.exists();os.chdir(ROOT/'Garnet');ledger.source_identity=lambda:{'scope':'Explicit CPU original native-build reader missing-parent-metadata failure and exact linked-parent lookup recovery; no native/GPU mutation and protected unrelated untracked excluded','owned_source_sha256':pins};artifacts=[ROOT/n for n in pins]+[ROOT/'results/q128-canonical-build-attempt1-evidence.zip']
    assert ledger.run([sys.executable,str(M/'verify_q128_canonical_native_build.py'),'--proofs',str(ROOT/'results')],out/'original-failure-ledger','correctness','Preserve actual original reader KeyError for parent metadata omitted from new archive; original native build outcome0 remains unchanged',artifacts)==1
    assert ledger.run([sys.executable,str(M/'verify_q128_canonical_native_build_parent.py'),'--proofs',str(ROOT/'results')],out/'replay-ledger','correctness','Replay identical native build/contracts/restoration gates, read parent50 source metadata from its already SHA-verified linked original archive; no native or GPU rerun',artifacts)==0
    print(json.dumps(dict(status='actual_Q128_native_build_recovered_reader_only',proof_sha256=sha(ROOT/'results/q128-canonical-build-attempt1-verified.json'))))
if __name__=='__main__':main()
