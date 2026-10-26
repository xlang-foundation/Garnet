"""Actual embedded-Python compatibility; fake native imports and IO only."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import builtins,json,runpy,sys

root,out=map(Path,sys.argv[1:3]);assert not out.exists() and out.resolve().is_relative_to((root/'results').resolve())
out.mkdir();sys.path.insert(0,str(root/'Garnet'))
from test2026.cuda.qwen38_5090.native_cuda_phase_entry import EntryLog,execute,IMPORTS
old=list(sys.argv)
try:
    sys.argv=[str(Path(__file__).with_name('probe_native_cuda_trace_embedded_cpu.py')),str(root),str(out/'trace-fixture')]
    runpy.run_path(sys.argv[0],run_name='__main__')
finally:sys.argv=old
real=builtins.__import__
def fake(name,*args,**kwargs):
    if name=='garnet':return SimpleNamespace()
    if name=='xModel.qwen38.vl_27b.model':
        return SimpleNamespace(create_cuda_bf16_low_batch_study_decoder=None,create_cuda_vision=None,prepare_cuda_image=None)
    return real(name,*args,**kwargs)
source='\n'+IMPORTS[0][0]+'\n'+IMPORTS[1][0]+'\nraise RuntimeError("embedded fake native error")\n'
log=EntryLog(out/'entry.jsonl','ordinary')
try:
    with patch('builtins.__import__',side_effect=fake):execute(source,Path(__file__),out/'entry-source.py.txt',log)
except RuntimeError as error:assert str(error)=='embedded fake native error'
else:raise AssertionError('Original fake error was suppressed')
rows=[json.loads(v) for v in (out/'entry.jsonl').read_bytes().splitlines()]
assert [r['stage'] for r in rows]==['python_entry_started','instrumented_source_retained','before_garnet_import',
    'garnet_import_returned','before_model_import','model_import_returned','worker_raised']
assert rows[-1]['exception']['type']=='RuntimeError' and 'embedded fake native error' in rows[-1]['exception']['traceback']
report={'status':'embedded_python_entry_cpu_compatibility_passed','gpu_executed':False,'actual_native_import':False,
        'entry_events':len(rows),'scope':'Real embedded runtime, intercepted synthetic imports and old fake-IO trace compatibility. No Garnet/model/GPU calls.'}
(out/'report.json').write_bytes(json.dumps(report,indent=2).encode());print('EMBEDDED_ENTRY_CPU_COMPATIBILITY_PASS',flush=True)
