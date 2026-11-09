"""Compose pinned existing controls and the separate merged-KV candidate."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

source=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();here=Path(__file__).resolve().parent
assert not out.exists(), 'Prepare a fresh directory.'
subprocess.run([sys.executable,str(here.parent/'mistral_contiguous_softmax/prepare_shader.py'),str(source),str(out)],check=True)
original=(out/'combined2.metal').read_bytes()
assert hashlib.sha256(original).hexdigest()=='b3fb8080347141b8384a0dfe128657095353fc2f0fb30b9a1c15defe62da249a'
policy=here/'chunk_mps_contiguous_policy.h'
assert hashlib.sha256(policy.read_bytes()).hexdigest()=='d3d3859aac860baed2a6bf2fc44244ac86b4da72767cc89c795cca994b1ebaee'
candidate=(here/'merged-kv1.metal').read_bytes();combined=original+candidate
(out/'combined1.metal').write_bytes(combined)
(out/'merged-bindings1.json').write_text(json.dumps(dict(original_combined_sha256=hashlib.sha256(original).hexdigest(),candidate_sha256=hashlib.sha256(candidate).hexdigest(),combined_sha256=hashlib.sha256(combined).hexdigest(),native_integration=False),indent=2)+'\n')
print(out/'combined1.metal')
