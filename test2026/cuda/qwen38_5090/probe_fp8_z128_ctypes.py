"""Embedded CPU-only ctypes capability check; never import Garnet or load GPU DLLs."""
from pathlib import Path
import ctypes,_ctypes,hashlib,json,sys
root,target=map(Path,sys.argv[1:3]);assert not target.exists()
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert root==Path('D:/CantorAI/5090-study-20261023') and target.parent==root/'results'
assert callable(ctypes.CDLL) and callable(ctypes.WinDLL) and ctypes.sizeof(ctypes.c_void_p)==8
assert 'garnet' not in sys.modules and 'tensor' not in sys.modules
result={'status':'embedded_ctypes_CPU_import_passed','python':sys.version,'executable':sys.executable,'pointer_bytes':ctypes.sizeof(ctypes.c_void_p),'ctypes_module':ctypes.__file__,'extension_module':_ctypes.__file__,'extension_sha256':sha(Path(_ctypes.__file__)),'runtime_host_sha256':sha(root/'build/bin/xlang3.exe'),'baseline_library_sha256':sha(root/'build/bin/garnet.dll'),'probe_sha256':sha(Path(__file__)),'gpu_library_explicitly_loaded':False,'garnet_imported':False,'model_or_performance_qualified':False}
raw=json.dumps(result,indent=2).encode();target.write_bytes(raw);print(json.dumps({'status':result['status'],'report_sha256':hashlib.sha256(raw).hexdigest()}))
