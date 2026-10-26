"""Actual embedded native-package CPU invocation, never invokes the helper or CUDA."""
from pathlib import Path
import hashlib,json,sys
root,out=map(Path,sys.argv[1:3]);job=root/'results/fp8-z128-native-package-import-attempt1';assert out==job/'report.json' and not out.exists()
sys.path.insert(0,str(job/'build'))
import fp8_z128_context_native as native
abi=native.abi();assert type(abi) is int and abi>0
paths=['C:/outside.json','D:/CantorAI/5090-study-20261023/results/fp8-z128-native-package-audit-attempt1/../escape.json','']
codes=[native.observe(path) for path in paths];assert codes==[20,20,20]
result={'status':'embedded_native_package_CPU_invocation_passed','abi':abi,'rejected_paths':paths,'rejection_codes':codes,'cpu_native_calls':4,'native_module_file':getattr(native,'__file__',None),'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runtime_host_sha256':hashlib.sha256((root/'build/bin/xlang3.exe').read_bytes()).hexdigest(),'observer_DLL_loaded_by_adapter':False,'helper_invoked':False,'gpu_model_executed':False,'model_or_performance_qualified':False,'scope':'Existing XLang package Load/BuildAPI/Create and member invocation with ABI query plus3outside-path guards, all before LoadLibrary/observer/CUDA. No ctypes/libffi/SDK mutation.'}
out.write_bytes(json.dumps(result,indent=2).encode());print(json.dumps(result))
