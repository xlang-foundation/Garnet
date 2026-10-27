"""Embedded Python import preflight only; never request Garnet or GPU resources."""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from test2026.cuda.qwen38_5090.fa3_projection_witness_layout import STAGES,check_graph,weight_metadata
assert STAGES==(('normalized',5120),('key',1024),('value',1024)) and 'numpy' not in sys.modules
assert len(hashlib.sha256(b'fa3-metadata-preflight').hexdigest())==64
target=Path(sys.argv[1]);assert not target.exists()
report=dict(status='embedded_FA3_metadata_import_passed',numpy_imported=False,python_version=sys.version,GPU_or_Garnet_requested=False)
target.write_bytes(json.dumps(report,indent=2).encode());print(json.dumps(report))
