"""Bind the explicit owned inventory before the unchanged B4 reference owner."""
from pathlib import Path
import copy,hashlib,json,sys

def bound_identity(p):
    key='Garnet/test2026/cuda/qwen38_5090/'+Path(__file__).name
    assert p['files'][key]==p['new_files'][key]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {'scope':'Explicit owned fresh B4 reference inputs; unrelated untracked files excluded','owned_source_sha256':copy.deepcopy(p['files'])}

def main():
    root=Path('D:/CantorAI/5090-study-20261023');m=root/'Garnet/test2026/cuda/qwen38_5090'
    assert Path(__file__).resolve().parent==m.resolve();sys.path.insert(0,str(m))
    p=json.loads((m/'evidence/bf16-batch8-external-reference-preparation.json').read_bytes())
    identity=bound_identity(p)
    import run_logged
    run_logged.source_identity=lambda:copy.deepcopy(identity)
    from run_bf16_batch8_external_reference import main as original_main
    return original_main()

if __name__=='__main__':raise SystemExit(main())
