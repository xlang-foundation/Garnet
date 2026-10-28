"""Accept the qualified89 model members plus exactly two study transport paths."""
from pathlib import PurePosixPath
import hashlib,zipfile
TRANSPORT={'test2026/cuda/qwen38_5090/native_http_bridge.py','test2026/cuda/qwen38_5090/native_worker_io.py'}

def load_bundle(bundle,model,expected):
    assert len(model)==89 and len(expected)==91 and set(expected)-set(model)==TRANSPORT
    assert all(expected[n]==h for n,h in model.items())
    assert all(n.startswith(('src/','xModel/')) for n in model)
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==91 and set(z.namelist())==set(expected)
        result={n:z.read(n) for n in expected}
    for n,raw in result.items():
        p=PurePosixPath(n)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in n and ':' not in n
        assert n in model or n in TRANSPORT
        assert hashlib.sha256(raw).hexdigest()==expected[n]
    return result
