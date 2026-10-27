"""Embedded import preflight; do not request Garnet or GPU resources."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.kv128_prefix_cache_witness import PREFIXES,parts,check_graphs
assert PREFIXES==(128,256,512,1024,2048,3968) and len(parts())==2 and 'numpy' not in sys.modules
assert len(hashlib.sha256(b'KV128-prefix-stdlib-metadata').hexdigest())==64
p=Path(sys.argv[1]);assert not p.exists();report=dict(status='embedded_KV128_prefix_metadata_passed',numpy_imported=False,GPU_or_Garnet_requested=False,python_version=sys.version);p.write_bytes(json.dumps(report,indent=2).encode());print(json.dumps(report))
