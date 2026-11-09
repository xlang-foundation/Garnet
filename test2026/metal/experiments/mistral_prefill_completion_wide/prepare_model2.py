"""Prepare untimed full-model control/candidate state comparison."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;N=M/'mistral-fused-mlp-native1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(ROOT/'build-audit2.json')['passed']
text=(N/'worker-model1.py').read_text();changes=[]
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:80],text.count(a));text=text.replace(a,b);changes.append(dict(old=a,new=b))
edit("        if chunked and not policy.set_prefill_chunk(512 if wide_owner else chunk):raise ValueError('Native prefill chunk configuration rejected')",'''        if chunked and not policy.set_prefill_chunk(512 if wide_owner else chunk):raise ValueError('Native prefill chunk configuration rejected')
        completion_first=profile['prefill_completion_priority']
        if type(completion_first) is not bool or 'half_attention_maximum_chunk=512' not in backend:
            raise ValueError('Explicit completion policy and qualified wide half backend required')
        if not policy.set_prefill_completion_priority(completion_first):raise ValueError('Native completion policy configuration rejected')
        if json.loads(policy.status_json())['prefill_completion_priority']!=completion_first:raise ValueError('Native completion policy status differs')
        emit(dict(event='native_prefill_completion_policy',enabled=completion_first,owner_limit=1 if completion_first else batch,chunk=chunk))''')
edit('            if diagnostic_forward[0] in [1,16,18,25]:', '            if not profile[\'prefill_completion_priority\'] and diagnostic_forward[0] in [1,16,18,25]:')
edit('        def deliver(request,token,stage):', '''        def capture_request_state(request,phase):
            row=request['row'];kv=[]
            for layer in range(32):
                pair=decoder.session.snapshot(layer)
                kv.append([decoder.io.verification_f32_sha256(decoder.io.rows(value,row,1)) for value in pair])
            hidden=decoder.io.rows(decoder.session.diagnostic_final_hidden(),row,1)
            emit(dict(event='diagnostic_request_state',id=jobs[request['id']]['id'],native_request=request['id'],row=row,epoch=request['epoch'],phase=phase,
                consumed_tokens=request['context_length'],kv=kv,hidden=decoder.io.verification_f32_sha256(hidden),hidden_values=hidden.tolist(),scope='Complete single leased row across32-layer canonical FP32 KV and final hidden'))
        def deliver(request,token,stage):''')
edit("            finished=token in stops or len(job['generated'])>=job['limit']",'''            finished=token in stops or len(job['generated'])>=job['limit']
            if stage=='prefill' or finished:capture_request_state(request,'prefill' if stage=='prefill' else 'final')''')
p=ROOT/'worker-model2.py';assert not p.exists();p.write_text(text);compile(text,str(p),'exec')
profiles={}
for label in ['control','wide']:
    profile=read(N/'profile-model1.json') if label=='control' else read(ROOT/'profile1.json')
    profile.update(profile_id='mistral-wide-half-model-'+label+'2',prefill_completion_priority=(label=='wide'),diagnostic_fixture=read(N/'profile-model1.json')['diagnostic_fixture'],memory_inventory_path=str(ROOT/('engine-inventory-'+label+'2.json')),scope='Untimed full-model state comparison; native snapshot copying, not performance measurement.')
    p=ROOT/('profile-model-'+label+'2.json');assert not p.exists();p.write_text(json.dumps(profile,indent=2)+'\n');profiles[label]=str(p)
(ROOT/'model-preparation2.json').write_text(json.dumps(dict(passed=True,original_worker_sha256=sha(N/'worker-model1.py'),new_worker_sha256=sha(ROOT/'worker-model2.py'),changes=changes,profiles=profiles,source_unchanged=True,performance_measurement=False,goal_complete=False),indent=2)+'\n')
print(json.dumps(dict(prepared=True,worker_sha256=sha(ROOT/'worker-model2.py'),profiles=profiles)),flush=True)
