"""Embedded stdlib preflight before requesting Garnet/model/GPU resources."""
from pathlib import Path
import ast,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.full_chunk_stage_layout import STAGES,stage_offsets
assert len(STAGES)==10 and sum(w for n,w in STAGES)==51200 and stage_offsets(128,17)[2]==('qg',1310737,1572864)
from test2026.cuda.qwen38_5090.q128_model_layout import reference_profile
assert reference_profile(128,True)=='baseline' and reference_profile(128,False)=='native' and reference_profile(32,False)=='baseline'
assert 'numpy' not in sys.modules and len(hashlib.sha256(b'full-chunk-stage-stdlib').hexdigest())==64
for n in ('full_chunk_stage_chunk.py','full_chunk_stage_program.py','probe_q128_model_admission.py','q128_model_layout.py'):ast.parse((Path(__file__).resolve().parent/n).read_bytes())
p=Path(sys.argv[1]);assert not p.exists();r=dict(status='embedded_full_chunk_stage_metadata_passed',numpy_imported=False,GPU_or_Garnet_requested=False,python_version=sys.version);p.write_bytes(json.dumps(r,indent=2).encode());print(json.dumps(r))
