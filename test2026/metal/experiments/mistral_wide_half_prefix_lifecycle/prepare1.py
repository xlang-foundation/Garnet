"""Prepare native wide-prefix return/edit regression with three live survivors."""
from pathlib import Path
import copy,hashlib,json,shutil,sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
assert len(sys.argv)==2, 'usage: prepare1.py FRESH-DIRECTORY-under-mistral7b-local-coder'
ROOT=Path(sys.argv[1]).resolve();assert not ROOT.exists();ROOT.mkdir()
M=ROOT.parent;J=M/'mistral-prefill-completion-wide1';P=M/'mistral-fused-mlp-prefix-http1';SG=M/'mistral-fused-mlp-serving1'
read=lambda p:json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    assert not p.exists(),str(p);p.write_text(json.dumps(v,indent=2)+'\n')
assert sha(J/'provenance1.json')=='3dd125f8f9ed9bef0a944cdb14eca7ff0f7af3b0bc19f818fc884763d99a9411'
assert sha(P/'provenance1.json')=='d9abf05f94f2c8f92c066717eee3ca5824ed39766c7412e8923d0bf2ab4ed96c'
assert sha(SG/'provenance1.json')=='0ae9d716752330deee4f5d5ebc4eabf45825e5f03963ff0ad8fc14262d265a9a'
assert read(J/'model-audit2.json')['passed'] and read(J/'admission-audit3.json')['passed']
tools=ROOT/'tools1';assert not tools.exists();tools.mkdir()
for p in (J/'tools2').glob('*.py'):shutil.copyfile(p,tools/p.name)
p=tools/'local_coder_batch_worker.py';text=p.read_text();original=text;changes=[]
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:90],text.count(a));text=text.replace(a,b);changes.append(dict(old=a,new=b))
edit('decoder=None;policy=None;jobs={};keys=set()','decoder=None;policy=None;jobs={};keys=set();observed_leases={}')
observer=(P/'tools1/local_coder_batch_worker.py').read_text()
capture=observer[observer.index('        def capture_target_state('):observer.index('        def deliver(')]
capture=capture.replace("state=dict(kv=kv,hidden=decoder.io.verification_f32_sha256(decoder.io.rows(hidden,row,1)))","row_hidden=decoder.io.rows(hidden,row,1)\n            state=dict(kv=kv,hidden=decoder.io.verification_f32_sha256(row_hidden),hidden_values=row_hidden.tolist(),consumed_tokens=request['context_length'])",1)
edit('        def deliver(request,token,stage):',capture+'        def deliver(request,token,stage):')
edit("            finished=token in stops or len(job['generated'])>=job['limit']", "            finished=token in stops or len(job['generated'])>=job['limit']\n            if stage=='prefill':capture_target_state(request,'prefill')\n            if finished:capture_target_state(request,'final')")
survivor=observer[observer.index('        def survivor_state():'):observer.index('        while not shutdown.is_set():',observer.index('        def survivor_state():'))]
edit('        while not shutdown.is_set():\n            if not jobs:',survivor+'        while not shutdown.is_set():\n            if not jobs:')
edit("            tick=json.loads(policy.next_json());plans=tick['batches']",'''            live={identity:lease for identity,lease in observed_leases.items() if identity in jobs}
            pending=[identity for identity in jobs if identity not in observed_leases]
            observe=bool(live) and len(live)<batch and bool(pending)
            before=survivor_state() if observe else None
            tick=json.loads(policy.next_json());plans=tick['batches']
            if observe:
                after=survivor_state()
                if before!=after:raise ValueError('New lease/restore changed a live native survivor')
                emit(dict(event='native_survivor_restore_witness',pending_requests=pending,before=before,after=after))
            for plan in plans:
                for request in plan['requests']:observed_leases[request['id']]=(request['row'],request['epoch'])''')
p.write_text(text);compile(text,str(p),'exec')
profile=read(J/'profile2.json');profile.update(profile_id='mistral-wide-half-prefix-live1',trace_stage_latency=True,prefix_entries=128,prefix_host_bytes=2560*1024*1024,scope='Untimed native wide-profile prefix return/append/edit with three already-prefilled live survivors. State copying perturbs timing; no speed or coding claim.')
profile['prefix_identity'][1]='mistral-wide-half-prefix-live1';profile['prefix_identity'][4]='native-wide-half-prefix-lifecycle1'
save(ROOT/'profile-template1.json',profile)
model=Path(read(SG/'fixture1.json')['model']);metadata=read(model/'tokenizer_config.json');template=metadata['chat_template']
if isinstance(template,list):template={e['name']:e['template'] for e in template}
if isinstance(template,dict):template=template['default']
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
def fail(message):raise ValueError(message)
env.globals['raise_exception']=fail;template=env.from_string(template);tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
def make(label,body):
    body=copy.deepcopy(body);rendered=template.render(messages=body['messages'],add_generation_prompt=True,enable_thinking=False,tools=None,bos_token=metadata['bos_token'],eos_token=metadata['eos_token'])
    ids=tokenizer.encode(rendered,add_special_tokens=False).ids
    assert 0<len(ids) and len(ids)+body['max_tokens']<=4096
    return dict(label=label,body=body,rendered_prompt=rendered,prompt_ids=ids,prompt_tokens=len(ids))
bodies=read(SG/'cases1/p2048-c4-o128.json');plans=read(SG/'serving-manifest1.json')['plans'];workload=read(Path(next(p['workload'] for p in plans if p['id']=='p2048-c4-o128')))
for i,b in enumerate(bodies):assert make(str(i),b)['prompt_ids']==workload['requests'][i]['token_ids']
entries=[]
for i,(label,limit) in enumerate([('B',192),('C',224),('D',256)],1):
    body=copy.deepcopy(bodies[i]);body['max_tokens']=limit;entries.append(make(label,body))
a=copy.deepcopy(bodies[0]);a['max_tokens']=8;entries.append(make('A',a))
e=copy.deepcopy(read(P/'fixture2.json')['requests'][4]['body']);e['max_tokens']=8;entries.append(make('E',e))
entries.append(make('A-return',a));append=copy.deepcopy(a);append['messages'][0]['content']+='\nAdditional requirement: include a short comment describing the complexity.';entries.append(make('append',append))
tail=copy.deepcopy(a);value=tail['messages'][0]['content'];assert value.count('Return only code.')==1;tail['messages'][0]['content']=value.replace('Return only code.','Return only Python code.');entries.append(make('changed-tail',tail))
head=copy.deepcopy(a);head['messages'][0]['content']='Changed request. '+head['messages'][0]['content'];entries.append(make('changed-head',head))
source=next(e['prompt_ids'] for e in entries if e['label']=='A')
def lcp(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y:break
        n+=1
    return n
prefixes={e['label']:lcp(source,e['prompt_ids']) for e in entries[5:]}
expected={name:(len(source) if name=='A-return' else count//256*256) for name,count in prefixes.items()}
assert len(source)==2048 and expected['A-return']==2048 and expected['append']>=1536 and expected['changed-tail']>=1536 and expected['changed-head']==0
save(ROOT/'fixture1.json',dict(scope='Synthetic cache correctness arrivals built from frozen long-prompt coding workloads; no coding-quality or speed admission.',model=str(model),requests=entries,seed_order=['B','C','D'],after_seed_first_token='D',newcomer_chain=['A','E','A-return','append','changed-tail','changed-head'],common_prefix_tokens_to_A=prefixes,expected_cached_tokens=expected,expected_total_output_limit=sum(e['body']['max_tokens'] for e in entries)))
save(ROOT/'bindings1.json',dict(passed=True,wide_native_root=str(J),native_provenance_sha256=sha(J/'provenance1.json'),source=str(J/'source2'),source_bindings=str(J/'source-bindings2.json'),source_binding_sha256=sha(J/'source-bindings2.json'),library=str(J/'library2'),library_sha256=sha(J/'library2/libgarnet.dylib'),model=str(model),model_files_sha256=read(SG/'plan1.json')['model_files_sha256'],native_model_gate_sha256=sha(J/'model-audit2.json'),native_admission_gate_sha256=sha(J/'admission-audit3.json'),fixture_sha256=sha(ROOT/'fixture1.json'),worker_sha256=sha(tools/'local_coder_batch_worker.py')))
save(ROOT/'preparation1.json',dict(passed=True,original_worker_sha256=sha(J/'tools2/local_coder_batch_worker.py'),observer_source_sha256=sha(P/'tools1/local_coder_batch_worker.py'),worker_sha256=sha(tools/'local_coder_batch_worker.py'),changes=changes,native_source_changed=False,new_compilation_required=False,independent_tokenizer_matched_frozen_prompts=True,three_survivors_ready_before_newcomer_native_plan_required=True,prefix_host_budget_bytes=profile['prefix_host_bytes'],performance_measurement=False,goal_complete=False))
print(json.dumps(dict(prepared=True,prompt_tokens={e['label']:e['prompt_tokens'] for e in entries},lcp=prefixes,expected_cached_tokens=expected,output_limit=sum(e['body']['max_tokens'] for e in entries))),flush=True)

expected=json.loads((HERE/"qualification1.json").read_text())["reproduced_input_sha256"]
for name,digest in expected.items():assert sha(ROOT/name)==digest,name
save(ROOT/"input-replay1.json",dict(passed=True,files_sha256=expected,inference_executed=False,native_source_changed=False,performance_measurement=False))
