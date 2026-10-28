"""Capture full current-native synthetic MTP outputs; retain strict failures."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import garnet
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram

manifest, geometry, prefix = sys.argv[1:]
prefix = Path(prefix)
output = Path(str(prefix)+'.json')
assert not output.exists()
cases = json.loads(Path(manifest).read_text())
result = {'scope':'Current-native full synthetic MTP outputs, no timing/quality acceptance.',
          'strict_numerical_gate_passed':False, 'cases':[], 'diagnostic_complete':False, 'pid':os.getpid()}
def cpu(t):
    return garnet.tensor_to_cpu(t).tolist()
def raw(values):
    return struct.pack('<'+str(len(values))+'f', *values)
def read(spec):
    blob = Path(spec['path']).read_bytes()
    return list(struct.unpack('<'+str(len(blob)//4)+'f', blob))
def save():
    output.write_text(json.dumps(result, indent=2))
for index, c in enumerate(cases):
    if ('-'+geometry+'-') not in Path(c['checkpoint']).name:
        continue
    io = garnet.cuda_tensor_io()
    io.open(c['checkpoint'])
    b,t,h = c['batch'],c['chunk'],c['config']['hidden_size']
    shape = [b,c['config']['num_key_value_heads'],c['capacity'],c['config']['head_dim']]
    key,value = io.zeros(shape),io.zeros(shape)
    hidden = io.zeros([b,t,h])
    width = io.tensor_bytes(key)//4//b
    buffers = [io.zeros(s['shape']) for s in c['steps'][0]['inputs']]
    program = CUDAMTPProgram(c['checkpoint'],c['config'],b,c['capacity'],t)
    record = {'case':index, 'batch':b,'chunk':t,'runs':[]}
    result['cases'].append(record)
    canonical = {}
    try:
        for repeat, graph in enumerate((False,True,True)):
            io.zero(key);io.zero(value)
            program.graph.enable_cuda_graph(graph)
            for step_index, step in enumerate(c['steps']):
                host = [read(s) for s in step['inputs']]
                for buffer, values in zip(buffers,host):
                    io.update(buffer,values)
                before = [raw(cpu(key)),raw(cpu(value))]
                packed = program.run([buffers[0],buffers[1],key,value,*buffers[2:]])
                io.slice_into(packed,0,hidden)
                io.slice_into(packed,t*h,key)
                io.slice_into(packed,t*h+width,value)
                assert all(raw(cpu(buf)) == raw(values) for buf,values in zip(buffers,host))
                tensors = []
                for output_index,(name,tensor,spec) in enumerate(zip(('hidden','key','value'),(hidden,key,value),step['expected'])):
                    actual = cpu(tensor)
                    blob = raw(actual)
                    expected = read(spec)
                    assert len(actual) == len(expected)
                    import math
                    assert all(math.isfinite(x) for x in actual+expected)
                    if repeat == 0:
                        canonical[(step_index,name)] = blob
                    assert canonical[(step_index,name)] == blob, 'Capture/replay changed full output'
                    bad = [(i,x,y) for i,(x,y) in enumerate(zip(actual,expected)) if abs(x-y)>2e-5+2e-4*abs(y)]
                    path = Path(str(prefix)+f'-c{index}-r{repeat}-s{step_index}-{name}.f32')
                    with path.open('xb') as stream:
                        stream.write(blob)
                    tensors.append({'name':name,'path':str(path),'words':len(actual),
                        'sha256':hashlib.sha256(blob).hexdigest(),'strict_bad_count':len(bad),
                        'max_abs':max(abs(x-y) for x,y in zip(actual,expected)), 'first_bad':bad[:8]})
                    if output_index:
                        dim,capacity,kv = shape[3],shape[2],shape[1]
                        for row in range(b):
                            written = {int(host[3][row*t+j]) for j in range(t) if host[4][row*t+j]}
                            for head in range(kv):
                                for slot in range(capacity):
                                    if slot not in written:
                                        start = ((row*kv+head)*capacity+slot)*dim*4
                                        assert blob[start:start+dim*4] == before[output_index-1][start:start+dim*4]
                record['runs'].append({'repeat':repeat,'graph':graph,'step':step_index,'tensors':tensors})
                save()
        stats = json.loads(program.graph.stats_json())
        assert stats['graph_captures']>0 and stats['graph_replays']>=2 and stats['graph_capture_failures']==0
        record['graph_stats'] = stats
        save()
        print('MTP_STRICT_STAGE_NATIVE_CASE_COMPLETE',index,flush=True)
    finally:
        program.close();io.close()
assert len(result['cases']) == (5 if geometry=='2b' else 2)
result['strict_numerical_gate_passed'] = all(x['strict_bad_count']==0 for c in result['cases'] for r in c['runs'] for x in r['tensors'])
result['diagnostic_complete'] = True
result['full_capture_replay_outputs_exact'] = True
result['inactive_cache_and_input_bytes_preserved'] = True
save()
print('MTP_STRICT_STAGE_NATIVE_COMPLETE',geometry,'NUMERICAL_GATE',result['strict_numerical_gate_passed'],flush=True)
