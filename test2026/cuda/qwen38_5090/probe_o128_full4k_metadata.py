"""Embedded stdlib preflight before requesting Garnet/model/GPU resources."""
from pathlib import Path
import ast,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.full_chunk_stage_layout import STAGES,stage_offsets
assert len(STAGES)==10 and sum(w for n,w in STAGES)==51200 and stage_offsets(128,17)[2]==('qg',1310737,1572864)
from test2026.cuda.qwen38_5090.o128_model_layout import reference
assert reference(128,True,True)==('kv128-prefix-cache-attempt1','baseline',True) and reference(128,True,False)==('q128-model-admission-attempt1','enabled128',False) and reference(32,False,False)==('kv128-prefix-cache-attempt1','baseline',True)
from test2026.cuda.qwen38_5090.o128_full4k_model import validate_targets
graphs={str(t):{'native_fp8_m128_canonical_projections':48 if t==128 else 0,'native_fp8_z128_canonical_projections':0,'native_fp8_z128_reconstructed_projections':48 if t==128 else 0,'native_fp8_kv128_canonical_projections':32 if t==128 else 0,'native_fp8_q128_canonical_projections':16 if t==128 else 0,'native_fp8_o128_canonical_projections':16 if t==128 else 0,'native_full_chunk_stage_witnesses':0} for t in (4,128)}
assert validate_targets(graphs,'1')['physical_GDN48_reconstructedZ48_KV32_verified'] is True
assert 'numpy' not in sys.modules and len(hashlib.sha256(b'full-chunk-stage-stdlib').hexdigest())==64
for n in ('full_chunk_stage_chunk.py','full_chunk_stage_program.py','probe_o128_full4k_model.py','o128_full4k_model.py'):ast.parse((Path(__file__).resolve().parent/n).read_bytes())
p=Path(sys.argv[1]);assert not p.exists();r=dict(status='embedded_o128_full4k_metadata_passed',numpy_imported=False,GPU_or_Garnet_requested=False,python_version=sys.version);p.write_bytes(json.dumps(r,indent=2).encode());print(json.dumps(r))
