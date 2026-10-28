"""Actual learned2B MTP child graph: complete legacy/bundle parity and ownership."""
import hashlib,json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram
root=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]);assert not output.exists()
data=json.loads((root/'mtp-learned-inputs-2b-chunk16.json').read_text());c=data['config'];h=c['hidden_size'];t=data['chunk']
assert h==2048 and t==16 and data['batch']==1
reference=json.loads((root/'mtp-native-learned-2b-chunk16-with-kv.json').read_text())
def saved_reference(field):
    path=Path(reference[field])
    if not path.is_absolute():path=root.parents[1]/path
    path=path.resolve()
    assert path.parent==root and path.is_file(),('Saved learned reference outside fixed root or absent',field,str(path))
    return path.read_bytes()
reference_hidden=saved_reference('normalized_hidden_file')
reference_key=saved_reference('cache_key_file')
reference_value=saved_reference('cache_value_file')
io=garnet.cuda_tensor_io();io.open(data['checkpoint'])
state=[1,c['num_key_value_heads'],data['capacity'],c['head_dim']]
key,value=io.zeros(state),io.zeros(state);width=io.tensor_bytes(key)//4
hidden=io.zeros([1,t,h]);last=io.zeros([1,h])
buffers=[io.zeros(s['shape']) for s in data['steps'][0]['inputs']]
args=[buffers[0],buffers[1],key,value,*buffers[2:]]
program=CUDAMTPProgram(data['checkpoint'],c,1,data['capacity'],t)
program.graph.enable_cuda_graph(True);program.graph.enable_serial_graph_reuse(True)
graph=program.graph;bundle=None
def raw(tensor):
    values=garnet.tensor_to_cpu(tensor).tolist();return struct.pack('<'+str(len(values))+'f',*values)
def update(step):
    for buf,spec in zip(buffers,step['inputs']):
        blob=Path(spec['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==spec['sha256']
        io.update(buf,list(struct.unpack('<'+str(len(blob)//4)+'f',blob)))
baseline=[]
try:
    for step in data['steps']:
        update(step);packed=program.run(args)
        io.slice_into(packed,0,hidden);io.slice_into(packed,t*h,key);io.slice_into(packed,t*h+width,value)
        io.copy_last_chunk(hidden,buffers[4],last)
        baseline.append({name:raw(tensor) for name,tensor in [('hidden',hidden),('key',key),('value',value),('last',last)]})
    joined=b''.join(row['hidden'][:step['count']*h*4] for row,step in zip(baseline,data['steps']))
    assert joined==reference_hidden,'New default learned hidden differs from preserved native component'
    assert baseline[-1]['key']==reference_key
    assert baseline[-1]['value']==reference_value
    stats=json.loads(graph.stats_json());assert stats['contexts']==stats['graph_captures']==1 and stats['graph_capture_failures']==0
    bundle=garnet.cuda_execution_bundle();packed=bundle.add_graph(graph,args)
    bundle.slice_into(packed,0,hidden);bundle.slice_into(packed,t*h,key);bundle.slice_into(packed,t*h+width,value)
    bundle.copy_last_chunk(hidden,buffers[4],last);bundle.seal()
    runs=[]
    for repeat in range(2):
        io.zero(key);io.zero(value);io.zero(last)
        if repeat==1:program.close()
        for index,step in enumerate(data['steps']):
            update(step);bundle.run()
            if repeat==1 and index==len(data['steps'])-1:bundle.close()
            actual={name:raw(tensor) for name,tensor in [('hidden',hidden),('key',key),('value',value),('last',last)]}
            if actual!=baseline[index]:
                for name,blob in actual.items():
                    with Path(str(output)+f'-r{repeat}-s{index}-{name}.actual.f32').open('xb') as stream:stream.write(blob)
                    with Path(str(output)+f'-r{repeat}-s{index}-{name}.expected.f32').open('xb') as stream:stream.write(baseline[index][name])
                raise AssertionError(('Learned bundle complete state/output mismatch',repeat,index))
            runs.append({'repeat':repeat,'step':index,'count':step['count'],
                         'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in actual.items()}})
    assert json.loads(graph.stats_json())['execution_bundle_reservations']==0
    assert json.loads(bundle.stats_json())['submissions']==2*len(data['steps'])
    with output.open('x') as stream:json.dump({'all_controls_passed':True,'process_id':os.getpid(),'complete_default_learned_outputs_exact_preserved_native':True,
        'complete_bundle_outputs_and_caches_exact':True,'source_closed_before_second_replay':True,
        'contexts_before_reservation':1,'runs':runs,'final_stats':json.loads(bundle.stats_json()),
        'scope':'Actual checkpoint MTP child only, caller-prepared teacher-forced inputs/chunk16. No target verification loop, recursive serving acceptance, independent model quality, primary latency/memory or2x claim.'},stream,indent=2)
    print('CUDA_EXECUTION_BUNDLE_LEARNED_PASS',len(runs),flush=True)
finally:
    if bundle is not None:bundle.close()
    program.close();io.close()
