"""Learned hybrid native prefix capture/restore, independent of HTTP serving.

XLang args: native build, source, checkpoint folder, frozen study JSON, result.
Checks exact and appended prompts against frozen native output tokens; checks a
changed suffix against fresh execution, while another row retains its state.
This is a finite correctness and resource study, not a benchmark campaign.
"""
import sys,json,time
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet,tensor
from xModel.qwen38.vl_27b.metal_batch_study import create_decoder
folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)
with open(sys.argv[4]) as stream:study=json.load(stream)
cases=study['requests'];batch=2;chunk=8;width=config['text_config']['hidden_size'];capacity=study['capacity'];limit=study['output_tokens']
assert 'native_hybrid_token_prefix' in garnet.backend_info()
assert len(cases)==batch and limit==16 and all(len(case['expected_ids'])==limit for case in cases)
assert all(len(case['token_ids'])+limit<=capacity for case in cases)
a=cases[0]['token_ids'];b=cases[1]['token_ids'];boundary=study['checkpoint_tokens']
assert boundary%chunk==0 and 0<boundary<len(a)
decoder=None;graphs=[];epochs=[]
report=dict(status='running',scope=__doc__,backend=garnet.backend_info(),checks=[],capacity=capacity,batch=batch,prefill_chunk=chunk)
def save():
    with open(sys.argv[5],'w') as stream:stream.write(json.dumps(report,indent=2)+'\n')
def f32(values,shape):return tensor.tensor(values,shape=shape,dtype=tensor.float32)
def record(name,**data):report['checks'].append(dict(name=name,**data));save();print('native_prefix_check',name,flush=True)
def prefill(rows,starts,capture_boundary=False):
    longest=max(len(tokens) for tokens in rows)
    for offset in range(0,longest,chunk):
        values=[];ids=[];flags=[];slots=[]
        for t in range(chunk):values.append(decoder.io.embedding([rows[row][offset+t] if offset+t<len(rows[row]) else 0 for row in range(batch)]))
        for row in range(batch):
            for t in range(chunk):
                valid=offset+t<len(rows[row]);ids.append(rows[row][offset+t] if valid else 0)
                flags.append(1.0 if valid else 0.0);slots.append(float(starts[row]+offset+t) if valid else 0.0)
        positions=[slot for slot in slots for axis in range(3)]
        decoder.session.run_tokens(graphs,decoder.io.stack_tokens(values),f32(positions,[batch,chunk,3]),
            f32(slots,[batch,chunk]),f32(flags,[batch,chunk]),-1,epochs,ids)
        if capture_boundary and offset+chunk==boundary:
            assert decoder.session.cache_token_prefix(0,epochs[0]);record('capture_intermediate',tokens=boundary,stats=json.loads(decoder.session.token_prefix_stats_json()))
    return decoder.session.predict_rows([epochs[row] if rows[row] else 0 for row in range(batch)])
def generate(predictions,lengths,count,selected):
    generated=[[predictions[row]] if row in selected else [] for row in range(batch)]
    for index in range(1,count):
        ids=[generated[row][-1] if row in selected else 0 for row in range(batch)]
        slots=[float(lengths[row]+index-1) if row in selected else 0.0 for row in range(batch)]
        positions=[slot for slot in slots for axis in range(3)]
        flags=[1.0 if row in selected else 0.0 for row in range(batch)]
        predicted=decoder.session.run_tokens([layer['graph'] for layer in decoder.layers],decoder.io.embedding(ids),
            f32(positions,[batch,3]),f32(slots,[batch]),f32(flags,[batch]),0,epochs,ids)
        for row in selected:generated[row].append(predicted[row])
    return generated

def replace_a(identity):
    old=epochs[0];assert decoder.session.release_request_row(0,old)
    lease=decoder.session.claim_request_row(identity);assert lease[0]==0 and lease[1]!=old;epochs[0]=lease[1]
try:
    report['memory_before']=json.loads(garnet.memory_stats_json())
    decoder=create_decoder(folder+'/model.safetensors.index.json',config,capacity,batch,chunk)
    decoder.prefill_compile_ms=0
    assert decoder.session.configure_token_prefix(study['identity'],2,384*1024*1024)
    graphs=decoder._resident_prefill_graphs(chunk)
    epochs=[decoder.session.claim_request_row(row+1)[1] for row in range(batch)]
    predicted=prefill([a,b],[0,0],True)
    assert decoder.session.cache_token_prefix(0,epochs[0]);record('capture_full',tokens=len(a),stats=json.loads(decoder.session.token_prefix_stats_json()))
    generated=generate(predicted,[len(a),len(b)],limit,[0,1]);assert generated==[case['expected_ids'] for case in cases]
    record('original_frozen_outputs',generated_ids=generated)
    b_prediction=generated[1][-1]
    replace_a(100)
    assert decoder.session.restore_token_prefix(0,epochs[0],a)==len(a)
    predicted=decoder.session.predict_rows(epochs);assert predicted[1]==b_prediction
    restored=generate(predicted,[len(a),len(b)],limit,[0]);assert restored[0]==cases[0]['expected_ids']
    assert decoder.session.predict_rows([0,epochs[1]])==[-1,b_prediction]
    record('exact_prompt_and_other_row_isolation',generated_ids=restored[0])
    replace_a(101);appended=a+[cases[0]['expected_ids'][0]]
    assert decoder.session.restore_token_prefix(0,epochs[0],appended)==len(a)
    predicted=prefill([[appended[-1]],[]],[len(a),0])
    continued=generate(predicted,[len(appended),len(b)],limit-1,[0])
    assert continued[0]==cases[0]['expected_ids'][1:]
    record('appended_prompt',reused_tokens=len(a),generated_ids=continued[0])
    replace_a(102);changed=list(a);changed[boundary]=study['changed_token_id']
    assert changed[boundary]!=a[boundary]
    assert decoder.session.restore_token_prefix(0,epochs[0],changed)==boundary
    predicted=prefill([changed[boundary:],[]],[boundary,0])
    changed_restored=generate(predicted,[len(changed),len(b)],4,[0])[0]
    assert decoder.session.predict_rows([0,epochs[1]])==[-1,b_prediction]
    replace_a(103)
    predicted=prefill([changed,[]],[0,0]);changed_fresh=generate(predicted,[len(changed),len(b)],4,[0])[0]
    assert changed_fresh==changed_restored
    record('changed_suffix_matches_fresh',reused_tokens=boundary,generated_ids=changed_restored)
    report['stats']=json.loads(decoder.session.token_prefix_stats_json());assert report['stats']['entries']==2 and report['stats']['hits']==3
    report['memory_before_clear']=json.loads(garnet.memory_stats_json())
    assert decoder.session.clear_token_prefixes();assert json.loads(decoder.session.token_prefix_stats_json())['accounted_host_bytes']==0
    report['status']='prefix_study_passed_not_full_qualification'
except:
    report['status']='failed';raise
finally:
    if decoder is not None:decoder.close()
    decoder=None;graphs=[]
    report['memory_after_close']=json.loads(garnet.memory_stats_json());save();time.sleep(0.5)
    report['memory_after_close_quiescence']=json.loads(garnet.memory_stats_json());save();garnet.shutdown()
