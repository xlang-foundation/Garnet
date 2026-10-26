"""Load exclusively owned observer after a failure or in an isolated cold process."""
from pathlib import Path
import ctypes,hashlib,json
from .fp8_z128_context_audit import replay
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def observe(root,target):
    root=Path(root);target=Path(target);job=root/'results/fp8-z128-context-audit-attempt1';assert target.resolve().is_relative_to(job.resolve()) and not target.exists()
    receipt=json.loads((job/'audit-library.json').read_bytes());library=job/'build/context-audit.dll'
    assert sha(library)==receipt['library_sha256']
    dependency=root/'build/bin/cublasLt64_13.dll';assert sha(dependency)==receipt['cublaslt_sha256']
    cublas_dependency=ctypes.WinDLL(str(dependency))
    module=ctypes.CDLL(str(library));call=module.AuditZ128Context;call.argtypes=[ctypes.c_char_p];call.restype=ctypes.c_int
    code=call(str(target).encode('ascii'));assert code==0,('Observer report error',code)
    report=json.loads(target.read_bytes());path=Path(bytes.fromhex(report['cublaslt_module_path_hex']).decode('ascii'))
    assert sha(path)==receipt['cublaslt_sha256']
    validation=replay(report,receipt['expected_m32_opaque'])
    return {'library_sha256':receipt['library_sha256'],'report_sha256':sha(target),'report_file':target.name,'loaded_cublaslt_path':str(path),'loaded_cublaslt_sha256':sha(path),'validation':validation,'placement_scope':'Separate observedTU of actual pinned helper in same process/CUDA context. Additional ID queries and report IO only after helper return; not a direct hook inside Garnet plugin.'}
