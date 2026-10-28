"""Exact MTP bundle outputs, unchanged old runtime controls and owner retirement."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import garnet
from xModel.qwen35.vl_0_8b.cuda_mtp import CUDAMTPProgram
root=Path(sys.argv[1]).resolve();geometry=sys.argv[2];output=Path(sys.argv[3]);assert not output.exists()
cases=json.loads((root/'mtp-component-fixtures.json').read_text())
records=[]
def raw(t):
    values=garnet.tensor_to_cpu(t).tolist()
    return struct.pack('<'+str(len(values))+'f',*values)
def host(spec):
    blob=Path(spec['path']).read_bytes();return list(struct.unpack('<'+str(len(blob)//4)+'f',blob))
def reject(fn,label,errors):
    try:fn()
    except Exception as error:errors.append({'label':label,'message':str(error)})
    else:raise AssertionError('Invalid bundle operation accepted: '+label)
def save():
    output.write_text(json.dumps({'cases':records,'scope':'Exact synthetic MTP bundle/lifecycle controls only; no original numerical tolerance waiver, learned quality, primary timing or memory acceptance.'},indent=2))
def compare(actual,expected,label):
    if actual!=expected:
        for name,blob in actual.items():
            with Path(str(output)+f'-{label}-{name}.actual.f32').open('xb') as stream:stream.write(blob)
            with Path(str(output)+f'-{label}-{name}.expected.f32').open('xb') as stream:stream.write(expected[name])
        raise AssertionError('Complete bundle output mismatch: '+label)
for index,c in enumerate(cases):
    if ('-'+geometry+'-') not in Path(c['checkpoint']).name:continue
    b,t,h=c['batch'],c['chunk'],c['config']['hidden_size']
    io=garnet.cuda_tensor_io();io.open(c['checkpoint'])
    state=[b,c['config']['num_key_value_heads'],c['capacity'],c['config']['head_dim']]
    key,value=io.zeros(state),io.zeros(state);width=io.tensor_bytes(key)//4//b
    hidden=io.zeros([b,t,h]);last=io.zeros([b,h])
    buffers=[io.zeros(s['shape']) for s in c['steps'][0]['inputs']]
    args=[buffers[0],buffers[1],key,value,*buffers[2:]]
    program=CUDAMTPProgram(c['checkpoint'],c['config'],b,c['capacity'],t)
    program.graph.enable_cuda_graph(True);program.graph.enable_serial_graph_reuse(True)
    graph=program.graph;bundle=None;errors=[];baseline=[]
    try:
        cold=garnet.cuda_execution_bundle()
        reject(lambda:cold.add_graph(graph,args),'unwarmed capture',errors);cold.close()
        assert json.loads(graph.stats_json())['contexts']==1
        for repeat in range(2):
            io.zero(key);io.zero(value);io.update(last,[7.125]*(b*h))
            current=[]
            for step_index,step in enumerate(c['steps']):
                for buf,spec in zip(buffers,step['inputs']):io.update(buf,host(spec))
                packed=program.run(args)
                io.slice_into(packed,0,hidden);io.slice_into(packed,t*h,key);io.slice_into(packed,t*h+width,value)
                io.copy_last_chunk(hidden,buffers[4],last)
                tensors={name:raw(tensor) for name,tensor in [('hidden',hidden),('key',key),('value',value),('last',last)]}
                for name in ('hidden','key','value'):
                    path=root/f'qwen35-mtp-stage-v2-native-{geometry}-c{index}-r0-s{step_index}-{name}.f32'
                    assert tensors[name]==path.read_bytes(),('Default path differs from saved5EAD complete output',index,step_index,name)
                current.append(tensors)
            if repeat==0:baseline=current
            else:assert current==baseline
        stats=json.loads(graph.stats_json());assert stats['contexts']==stats['graph_captures']==1 and stats['graph_capture_failures']==0
        invalid=garnet.cuda_execution_bundle()
        reject(lambda:invalid.add_graph(graph,args[:-1]),'input count',errors)
        replaced=list(args);replaced[0]=io.clone(buffers[0])
        reject(lambda:invalid.add_graph(graph,replaced),'different pointer binding',errors);invalid.close()
        assert json.loads(graph.stats_json())['contexts']==1 and json.loads(graph.stats_json())['execution_bundle_reservations']==0
        before=[raw(x) for x in args+[hidden,last,packed]]
        bundle=garnet.cuda_execution_bundle();result=bundle.add_graph(graph,args)
        reject(lambda:bundle.run(),'unsealed execution',errors)
        reject(lambda:bundle.add_graph(graph,args),'duplicate graph',errors)
        reject(lambda:bundle.slice_into(result,-1,hidden),'negative slice',errors)
        reject(lambda:bundle.slice_into(result,0,result),'overlapping slice',errors)
        reject(lambda:bundle.copy_last_chunk(hidden,buffers[4],buffers[4]),'last output shape/alias',errors)
        bundle.slice_into(result,0,hidden);bundle.slice_into(result,t*h,key);bundle.slice_into(result,t*h+width,value)
        bundle.copy_last_chunk(hidden,buffers[4],last);bundle.seal()
        assert before==[raw(x) for x in args+[hidden,last,packed]],'Bundle construction advanced/mutated tensors'
        stats=json.loads(graph.stats_json());assert stats['contexts']==1 and stats['execution_bundle_reservations']==1
        reject(lambda:program.run(args),'direct reserved parent execution',errors)
        reject(lambda:graph.enable_cuda_graph(False),'reserved capture policy mutation',errors)
        reject(lambda:graph.enable_serial_graph_reuse(False),'reserved reuse policy mutation',errors)
        reject(lambda:bundle.slice_into(result,0,hidden),'sealed graph mutation',errors)
        repeat_records=[]
        for repeat in range(3):
            io.zero(key);io.zero(value);io.update(last,[7.125]*(b*h))
            if repeat==2:
                program.close()
                assert not graph.ready and json.loads(graph.stats_json())['execution_bundle_reservations']==1
            for step_index,step in enumerate(c['steps']):
                for buf,spec in zip(buffers,step['inputs']):io.update(buf,host(spec))
                before_inputs=[raw(buf) for buf in buffers]
                bundle.run()
                if repeat==2 and step_index==1:bundle.close()
                actual={name:raw(tensor) for name,tensor in [('hidden',hidden),('key',key),('value',value),('last',last)]}
                compare(actual,baseline[step_index],f'c{index}-r{repeat}-s{step_index}')
                assert before_inputs==[raw(buf) for buf in buffers],'Bundle mutated read-only request operands'
                repeat_records.append({'repeat':repeat,'step':step_index,'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in actual.items()}})
        assert json.loads(graph.stats_json())['execution_bundle_reservations']==0
        end=json.loads(bundle.stats_json());assert end['closed'] and end['submissions']==6 and end['reserved_model_contexts']==0
        reject(lambda:bundle.run(),'closed execution',errors)
        record={'case':index,'batch':b,'chunk':t,'full_default_outputs_exact_saved5ead':True,
            'no_state_advance_during_build':True,'contexts_before_reservation':1,
            'source_closed_before_final_two_runs':True,'bundle_closed_before_last_observation':True,
            'all_complete_bundle_outputs_exact':True,'original_strict_numerical_gate_unchanged':True,
            'errors':errors,'runs':repeat_records,'final_stats':end}
        records.append(record);save();print('CUDA_EXECUTION_BUNDLE_CASE_PASS',index,flush=True)
    finally:
        if bundle is not None:bundle.close()
        program.close();io.close()
assert len(records)==(5 if geometry=='2b' else 2)
output.write_text(json.dumps({'all_controls_passed':True,'process_id':os.getpid(),'cases':records,
    'scope':'Synthetic complete-output/ownership/capture/lifetime coverage only. All original strict numerical failures remain failed; no quality/performance/memory or2x acceptance.'},indent=2))
print('CUDA_EXECUTION_BUNDLE_CONTROLS_PASS',geometry,len(records),flush=True)
