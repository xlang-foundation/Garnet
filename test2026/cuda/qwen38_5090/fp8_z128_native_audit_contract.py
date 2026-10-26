"""CPU-only native observer receipt validation before importing or invoking a package."""
OBSERVER='results/fp8-z128-context-audit-attempt1/build/context-audit.dll'

def validate(receipt,observer_sha,package_sha,cublas_sha):
    assert type(receipt['abi']) is int and receipt['abi']==24 and receipt['observer_library']==OBSERVER
    for name,actual in (('library_sha256',observer_sha),('native_package_library_sha256',package_sha),('cublaslt_sha256',cublas_sha)):
        value=receipt[name];assert type(value) is str and len(value)==64 and bytes.fromhex(value).hex()==value and value==actual
    expected=receipt['expected_m32_opaque'];assert type(expected) is str and len(expected)==128 and bytes.fromhex(expected).hex()==expected
    files=receipt['observed_helper_files'];assert len(files)==2 and all(name.startswith('Garnet/test2026/cuda/qwen38_5090/native_fp8_z128_candidate/') for name in files)
    return {'abi':24,'observer_library':OBSERVER,'expected_m32_opaque':expected,'native_package_library_sha256':package_sha,'observer_library_sha256':observer_sha,'cublaslt_sha256':cublas_sha}
