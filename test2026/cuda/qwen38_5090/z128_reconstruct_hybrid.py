"""CUDA runtime origin proof for an SDK executable linked to the pinned static library."""
import re
def identity(report,exe,dependencies,linkmap):
    norm=lambda value:value.replace(chr(92),'/').lower()
    assert report['cuda_runtime_linkage']=='hybrid_loader' and report['cuda_runtime_status']==report['cuda_driver_status']==0
    assert (report['cuda_runtime_header_version'],report['cuda_runtime_version'],report['cuda_driver_version'])==(13020,13040,13040)
    assert type(report['named_cudart_DLL_present']) is bool
    paths={row['name']:norm(bytes.fromhex(row['path_hex']).decode()) for row in report['module_paths']};assert len(paths)==3
    assert paths['cublasLt64_13.dll']=='d:/cantorai/5090-study-20261023/build/bin/cublaslt64_13.dll' and paths['cudaGetDevice']==norm(exe)
    assert paths['nvcuda.dll'].startswith('c:/windows/') and paths['nvcuda.dll'].endswith('.dll')
    if isinstance(dependencies,bytes):dependencies=dependencies.decode('ascii',errors='replace')
    if isinstance(linkmap,bytes):linkmap=linkmap.decode('ascii',errors='replace')
    imported=[line.strip().lower() for line in dependencies.splitlines() if re.fullmatch(r'\s*[\w.-]+\.dll\s*',line,re.I)]
    assert 'cublaslt64_13.dll' in imported and 'kernel32.dll' in imported and not any('cudart' in name for name in imported)
    origin=[line.strip() for line in linkmap.splitlines() if 'cudaGetDevice' in line and 'cudart:generated_cuda_runtime_api_loader.obj' in line]
    assert origin,'Link map must identify cudaGetDevice contribution from cudart library'
    return {'status':'Windows_hybrid_CUDA_runtime_origin_replayed','CUDA_API_loader_origin':paths['cudaGetDevice'],'CUDA_runtime_version':report['cuda_runtime_version'],'CUDA_header_version':report['cuda_runtime_header_version'],'CUDA_driver_version':report['cuda_driver_version'],'CUDA_driver_module':paths['nvcuda.dll'],'cuBLASLt_module':paths['cublasLt64_13.dll'],'PE_imported_DLLs':imported,'linker_cudaGetDevice_origins':origin,'named_CUDART_DLL_present':report['named_cudart_DLL_present'],'runtime_origin_by_address_and_hybrid_loader_map':True,'numerical_or_model_or_performance_qualified':False}
