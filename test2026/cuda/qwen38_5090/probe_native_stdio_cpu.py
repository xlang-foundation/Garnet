"""Embedded-runtime stdio diagnosis only; no Garnet/model/GPU import or call."""
from pathlib import Path
import json,os,sys

out=Path(sys.argv[1]).resolve();root=Path('D:/CantorAI/5090-study-20261023/results').resolve()
assert out.is_relative_to(root) and not out.exists();out.mkdir()
rows=[];report={'status':'collecting','pid':os.getpid(),'operations':rows,'actual_native_model_executed':False,
    'scope':'CPU-only embedded Python stdio observations; failures are recorded, not suppressed in a model worker.'}
def save():(out/'report.json').write_bytes(json.dumps(report,indent=2).encode())
def attempt(name,call):
    row={'operation':name};rows.append(row);save()
    try:row.update(returned=repr(call()),completed=True)
    except BaseException as error:row.update(completed=False,exception_type=type(error).__name__,exception_repr=repr(error),
        errno=getattr(error,'errno',None),winerror=getattr(error,'winerror',None))
    save()
save()
attempt('os.write.stdout',lambda:os.write(1,b'{"probe":"os.write.stdout"}\n'))
attempt('sys.stdout.write',lambda:(sys.stdout.write('{"probe":"sys.stdout.write"}\n'),sys.stdout.flush()))
attempt('sys.__stdout__.write',lambda:(sys.__stdout__.write('{"probe":"sys.__stdout__.write"}\n'),sys.__stdout__.flush()))
def buffered():
    with os.fdopen(1,'wb',closefd=False) as stream:return (stream.write(b'{"probe":"fdopen.stdout"}\n'),stream.flush())
attempt('os.fdopen.stdout',buffered)
attempt('os.write.stderr',lambda:os.write(2,b'{"probe":"os.write.stderr"}\n'))
attempt('os.read.stdin',lambda:os.read(0,65536))
def file_output():
    with (out/'file-output.json').open('xb') as stream:return stream.write(b'{"probe":"owned.file","cpu_only":true}\n')
attempt('owned_file.write',file_output)
report.update(status='cpu_stdio_observations_collected',sys_stdout_type=type(sys.stdout).__name__,sys_stdin_type=type(sys.stdin).__name__);save()
