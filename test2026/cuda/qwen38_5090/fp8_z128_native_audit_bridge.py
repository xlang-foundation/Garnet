"""Invoke unchanged helper observer through the existing native package ABI, no ctypes."""
from pathlib import Path
import hashlib,importlib,json,sys
from .fp8_z128_context_audit import replay
from .fp8_z128_native_audit_contract import validate
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def observe(root,target):
    root=Path(root);target=Path(target);job=root/'results/fp8-z128-native-package-audit-attempt1';assert target.resolve().is_relative_to(job.resolve()) and not target.exists()
    receipt=json.loads((job/'audit-library.json').read_bytes());package=job/'build/fp8_z128_context_native.dll';observer=root/receipt['observer_library']
    validate(receipt,sha(observer),sha(package),sha(root/'build/bin/cublasLt64_13.dll'))
    sys.path.insert(0,str(job/'build'));native=importlib.import_module('fp8_z128_context_native')
    assert native.abi()==24;loaded=Path(native.library_path());assert loaded.resolve()==package.resolve() and sha(loaded)==receipt['native_package_library_sha256']
    code=native.observe(str(target));assert code==0,('Native observer report error',code)
    report=json.loads(target.read_bytes());path=Path(bytes.fromhex(report['cublaslt_module_path_hex']).decode('ascii'));assert sha(path)==receipt['cublaslt_sha256']
    validation=replay(report,receipt['expected_m32_opaque'])
    return {'library_sha256':receipt['library_sha256'],'report_sha256':sha(target),'report_file':target.name,'loaded_cublaslt_path':str(path),'loaded_cublaslt_sha256':sha(path),'native_package_library_path':str(loaded),'native_package_library_sha256':sha(loaded),'native_package_ABI':24,'validation':validation,'placement_scope':'Actual unchanged helper in separate observedTU, synchronous XLang native call in same process/thread after failure (or isolated cold process). Additional ID queries and IO after helper return; not a direct hook inside Garnet plugin.'}
