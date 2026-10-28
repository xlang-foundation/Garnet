"""CPU-only check of Python versus Windows native narrow-CRT environment.

Queries only one synthetic diagnostic variable. No model, CUDA runtime, or
Garnet import; no user environment inventory is recorded.
"""
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

key = 'GARNET_DIAGNOSTIC_CRT_ENV_PROBE'
assert key not in os.environ
crt = ctypes.CDLL('ucrtbase.dll')
crt.getenv.argtypes = [ctypes.c_char_p]
crt.getenv.restype = ctypes.c_char_p
crt._putenv_s.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
crt._putenv_s.restype = ctypes.c_int
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.GetEnvironmentVariableW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
kernel.GetEnvironmentVariableW.restype = ctypes.c_uint32


def snapshot():
    value = crt.getenv(key.encode('ascii'))
    buffer = ctypes.create_unicode_buffer(256)
    count = kernel.GetEnvironmentVariableW(key, buffer, len(buffer))
    return {'python': os.environ.get(key), 'ucrt_narrow': value.decode() if value is not None else None,
            'windows_process': buffer.value if count else None}


before = snapshot()
try:
    os.environ[key] = 'qwen35_probe_value'
    python_set = snapshot()
    assert crt._putenv_s(key.encode(), b'qwen35_probe_value') == 0
    narrow_set = snapshot()
finally:
    os.environ.pop(key, None)
    assert crt._putenv_s(key.encode(), b'') == 0
after = snapshot()
assert before == after == {'python': None, 'ucrt_narrow': None, 'windows_process': None}
assert narrow_set == {'python': 'qwen35_probe_value', 'ucrt_narrow': 'qwen35_probe_value', 'windows_process': 'qwen35_probe_value'}
output = Path(sys.argv[1])
assert not output.exists()
result = {'before': before, 'after_python_environ_set': python_set, 'after_explicit_narrow_crt_set': narrow_set,
    'after_cleanup': after, 'python_mutation_visible_to_narrow_crt': python_set['ucrt_narrow'] == 'qwen35_probe_value',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'process_id': os.getpid(), 'scope': 'CPU-only actual embedded Python and system UCRT/Windows environment; no GPU/model or proof of Garnet CRT linkage.'}
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
