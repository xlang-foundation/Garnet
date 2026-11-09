"""Prepare unchanged frozen C4 requests for native candidate admission."""
from pathlib import Path
import difflib,hashlib,json,shutil,socket
ROOT=Path(__file__).resolve().parent;M=ROOT.parent;SG=M/'mistral-fused-mlp-serving1';N=M/'mistral-fused-mlp-native1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    assert not p.exists(),str(p);p.write_text(json.dumps(v,indent=2)+'\n')
tools=ROOT/'tools1';assert not tools.exists();tools.mkdir()
for p in (SG/'tools1').glob('*.py'):shutil.copyfile(p,tools/p.name)
p=tools/'local_coder_batch_worker.py';text=p.read_text();original=text;changes=[]
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:70],text.count(a));text=text.replace(a,b);changes.append(dict(old=a,new=b))
edit("    chunked=prefill_schedule=='chunked'",'''    chunked=prefill_schedule=='chunked'
    completion_first=profile.get('prefill_completion_priority',False)
    if type(completion_first) is not bool or (completion_first and (family!='mistral_7b' or not chunked or bulk_chunks)):
        raise ValueError('Completion-first scheduling requires an explicit bounded Mistral chunk profile without bulk grouping')
    completion_capable='native_prefill_completion_priority_optional' in garnet.backend_info()
    if completion_first and not completion_capable:raise ValueError('Native completion-first scheduler extension required')''')
edit("        if bulk_chunks:\n            if wide_owner or not policy.set_bulk_prefill(bulk_chunks)",'''        if completion_capable:
            if not policy.set_prefill_completion_priority(completion_first):raise ValueError('Native prefill policy configuration rejected')
            status=json.loads(policy.status_json())
            if status['prefill_completion_priority']!=completion_first:raise ValueError('Native scheduler policy status differs')
            emit(dict(event='native_prefill_completion_policy',enabled=completion_first,owner_limit=1 if completion_first else batch,chunk=chunk))
        if bulk_chunks:
            if wide_owner or not policy.set_bulk_prefill(bulk_chunks)''')
p.write_text(text)
(ROOT/'worker1.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='a/local_coder_batch_worker.py',tofile='b/local_coder_batch_worker.py')))
for p in tools.glob('*.py'):compile(p.read_text(),str(p),'exec')
profile=read(SG/'wave-2-p2048-c4-o128/profile.json');profile.update(profile_id='mistral-completion-first-candidate1',prefill_completion_priority=True,scope='Native completion-first candidate admission; unchanged frozen2K/C4/128-out requests, not serving/coding qualification.')
save(ROOT/'profile1.json',profile)
spec=read(SG/'wave-2-p2048-c4-o128/engine.json')
command=[]
for arg in spec['argv']:
    if arg.startswith(str(SG/'tools1')+'/'):arg=str(tools/Path(arg).name)
    elif arg==str(N/'library2'):arg=str(ROOT/'library1')
    elif arg==str(N/'source2'):arg=str(ROOT/'source1')
    elif arg==str(SG/'wave-2-p2048-c4-o128/profile.json'):arg=str(ROOT/'profile1.json')
    command.append(arg)
with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
command[command.index('--port')+1]=str(port);command[command.index('--log-root')+1]=str(ROOT/'server-template1')
spec.update(engine='garnet-native-mistral-completion-first-candidate1',url='http://127.0.0.1:'+str(port),argv=command,scope=profile['scope'],source_commit='0f9f3c35f',native_library_sha256=None,source_manifest_sha256=sha(ROOT/'source-bindings1.json'),profile_sha256=sha(ROOT/'profile1.json'),prefill_completion_priority=True,experimental_native_source=True)
save(ROOT/'engine1.json',spec)
save(ROOT/'admission-plan1.json',dict(passed=True,source_worker_sha256=sha(SG/'tools1/local_coder_batch_worker.py'),worker_sha256=sha(tools/'local_coder_batch_worker.py'),worker_changes=changes,fixture=str(SG/'fixture1.json'),serving_manifest=str(SG/'serving-manifest1.json'),case='p2048-c4-o128',rounds=['cold','warm1','warm2'],requests=12,expected_output_tokens=1536,profile_sha256=sha(ROOT/'profile1.json'),engine_sha256=sha(ROOT/'engine1.json'),no_comparator_execution=True,serving_qualified=False,goal_complete=False))
print(json.dumps(dict(prepared=True,worker_sha256=sha(tools/'local_coder_batch_worker.py'),port=port)),flush=True)
