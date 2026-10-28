"""Explicit diagnostic-only writer for embedded Python under Windows Nsight.

Use the original request parser, text normalization, JSON bytes and write-loop
checks. Nsight invalidates os.write stdout in this embedded runtime, while the
actual sys.stdout text stream reaches the owned pipe. No global stdio patch.
"""
import sys
from .native_worker_io import requests,emit as _emit

def _write(raw):
    # Original emit uses ensure_ascii=True: character and byte counts coincide.
    count=sys.stdout.write(raw.decode('ascii'))
    sys.stdout.flush()
    return count

def emit(value):
    return _emit(value,write=_write)
