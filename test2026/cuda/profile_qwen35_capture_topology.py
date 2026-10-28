"""Export existing native capture DOTs without changing graph/model execution."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

root=Path('D:/CantorAI/work/qwen35-cuda')
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo));sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
from runtime.cuda_serving import Model
assert os.environ['GARNET_TRT_ALLOC_TRACE']=='1'
assert Path(os.environ['GARNET_TRT_CAPTURE_DOT_DIR']).resolve()==root
assert os.environ.get('GARNET_PROFILE_DECODE_LAYERS') is None
output=Path(sys.argv[2]).resolve()
inventory=output.with_suffix('.capture-inventory.json')
assert not inventory.exists()
def snapshot():
    return {p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in root.glob('capture-garnet-cuda-graph-*.dot')}
initial=snapshot()
original=Model.infer_batch
request_index=0
def traced(self,*args,**kwargs):
    global request_index
    index=request_index;request_index+=1
    assert os.environ['GARNET_TRT_ALLOC_TRACE']=='1'
    assert Path(os.environ['GARNET_TRT_CAPTURE_DOT_DIR']).resolve()==root
    before=snapshot()
    result=original(self,*args,**kwargs)
    after=snapshot()
    result['capture_topology_diagnostic']={'request_index':index,
        'new_dot_files_this_request':sorted(set(after)-set(before)),
        'changed_existing_dot_files_this_request':sorted(n for n in before if after.get(n)!=before[n]),
        'native_layer_profiler_enabled':False,'primary_latency_accepted':False,
        'scope':'Existing native capture trace/DOT export enabled from startup; file metadata only at request boundaries, no extra model/GPU calls. Not a timing or memory measurement.'}
    return result
try:
    Model.infer_batch=traced
    entry=Path(__file__).with_name('benchmark_qwen35_shared_mtp.py')
    sys.argv[0]=str(entry);sys.argv[3:]=['baseline']
    runpy.run_path(str(entry),run_name='__main__')
    assert request_index==28
finally:
    Model.infer_batch=original
final=snapshot()
assert all(final.get(name)==stat for name,stat in initial.items()),'Existing capture DOT overwritten'
new=sorted(set(final)-set(initial));assert new,'Native capture hook produced no DOTs'
records=[{'file':str(root/name),'bytes':final[name][0],
          'sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in new]
with inventory.open('x') as stream:
    json.dump({'all28_completed':True,'initial_dot_stat_guard':initial,'new_dots':records,
               'hashing_scope':'New DOTs hashed after benchmark returned and model closed.'},stream,indent=2)
print('QWEN35_CAPTURE_TOPOLOGY_EXPORT_COMPLETE',len(new),flush=True)
