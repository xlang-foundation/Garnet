"""Actual embedded import before requesting any Garnet or GPU resource."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.gdn_kv128_stage_layout import STAGES,layout,check_graph
assert len(STAGES)==11 and STAGES[-1]==('output_hidden',5120) and 'numpy' not in sys.modules
assert len(hashlib.sha256(b'GDN50-stdlib-metadata').hexdigest())==64
target=Path(sys.argv[1]);assert not target.exists()
report=dict(status='embedded_GDN50_metadata_import_passed',numpy_imported=False,GPU_or_Garnet_requested=False,python_version=sys.version)
target.write_bytes(json.dumps(report,indent=2).encode());print(json.dumps(report))
