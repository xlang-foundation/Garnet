"""Freeze bounded helper input from original user messages; target stays intact."""
from pathlib import Path
import ast,hashlib,json
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
R=Path(__file__).resolve().parent;M=R.parent;S=M/'mistral-fused-q4-decode-coder1';N=M/'mistral-fused-q4-decode-native1';SG=M/'mistral-fused-mlp-serving1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):
    p=R/n;assert not p.exists(),n;p.write_text(json.dumps(d,indent=2)+'\n')
assert not (R/'preparation1.json').exists()
inputs=read(S/'input1.json');inputs['output_limit']=128
plan=next(x for x in read(SG/'serving-manifest1.json')['plans'] if x['id']=='p2048-c1-o128')
body=read(Path(plan['cases']))[0];workload=read(Path(plan['workload']));request=workload['requests'][0]
assert len(body['messages'])==1 and body['messages'][0]['role']=='user'
draft_model=Path(inputs['draft_model']);config=read(draft_model/'tokenizer_config.json');tok=Tokenizer.from_file(str(draft_model/'tokenizer.json'));template=SandboxedEnvironment().from_string(config['chat_template'])
text=body['messages'][0]['content'];user_ids=tok.encode(text,add_special_tokens=False).ids
tail=tok.decode(user_ids[-192:],skip_special_tokens=False)
bounded=template.render(messages=[dict(role='user',content=tail)],add_generation_prompt=True,enable_thinking=False,tools=None)
full=template.render(messages=body['messages'],add_generation_prompt=True,enable_thinking=False,tools=None)
bounded_ids=tok.encode(bounded,add_special_tokens=False).ids;full_ids=tok.encode(full,add_special_tokens=False).ids
assert len(bounded_ids)<=256 and len(bounded_ids)+8<=inputs['draft_capacity']==512 and len(full_ids)+8>512
# This fixture explicitly puts its task after repetitive reference padding.
suffix=text[text.rindex('Task:'):];assert suffix in tail
sources=[];controls=[]
for wave in ['wave-1-p2048-c1-o128','wave-4-p2048-c1-o128']:
    p=SG/wave/'campaign/observed/server/worker-stdout.jsonl';sources.append(dict(path=str(p),sha256=sha(p)))
    for line in p.read_text().splitlines():
        e=json.loads(json.loads(line)['line'])
        if e['event']=='completed' and e['prompt_ids']==request['token_ids']:controls.append(e)
assert controls and all(c['generated_ids']==controls[0]['generated_ids'] and c['text']==controls[0]['text'] for c in controls)
assert len(controls[0]['generated_ids'])==128 and len(request['token_ids'])==2048
save('frozen-control1.json',dict(scope='Extracted existing frozen native C1 outputs; never consulted by proposer or verifier.',prompt_ids=request['token_ids'],generated_ids=controls[0]['generated_ids'],text=controls[0]['text'],sources=sources))
inputs['cases']=[dict(id='p2048-c1-o128',rendered=request['rendered_prompt'],prompt_ids=request['token_ids'],control=str(R/'frozen-control1.json'),draft_prompt=bounded)]
inputs['draft_context_policy']=dict(name='fixed_user_tail192_v1',user_token_limit=192,full_target_prompt_unchanged=True,draft_base_prompt_tokens=len(bounded_ids),append_only_already_generated_target_text=True)
save('input1.json',inputs)
source=(S/'worker1.py').read_text();text=source;changes=[]
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:90],text.count(a));text=text.replace(a,b);changes.append(dict(old=a,new=b))
edit("assert len(sys.argv) in (5,6)\noccupancy=sys.argv[5] if len(sys.argv)==6 else None\nassert occupancy in (None,'occupancy-control','occupancy-candidate')\nmode_names=['base'] if occupancy else ['base','scores','stop']", "assert len(sys.argv)==6\nexperiment=sys.argv[5];assert experiment in ('qualify','ordinary','bounded')\noccupancy='occupancy-control'\nmode_names=['ordinary','base'] if experiment=='qualify' else ['ordinary'] if experiment=='ordinary' else ['base']")
edit("assert inputs['capacity']==4096 and inputs['output_limit']==1024 and inputs['draft_token_limit']==8", "assert inputs['capacity']==4096 and inputs['output_limit']==128 and inputs['draft_token_limit']==8\nassert inputs['draft_context_policy']['name']=='fixed_user_tail192_v1' and inputs['draft_capacity']==512\nassert not fused_enabled")
edit("scope=__doc__,projection_occupancy=occupancy,fused_decode_enabled=fused_enabled)", "scope=__doc__,projection_occupancy=occupancy,fused_decode_enabled=fused_enabled,experiment=experiment)")
edit("   use_confidence=mode!='base';stop_threshold=inputs['confidence_threshold'] if mode=='stop' else None", "   use_confidence=False;stop_threshold=None")
edit("   prompt=case['prompt_ids'];assert target_tokenizer.encode(case['rendered'])==prompt", "   prompt=case['prompt_ids'];assert len(prompt)==2048 and target_tokenizer.encode(case['rendered'])==prompt\n   assert len(draft_tokenizer.encode_ex(case['draft_prompt'],False))==inputs['draft_context_policy']['draft_base_prompt_tokens']")
edit("    prefill_seconds=prefill_seconds,decode_seconds=elapsed", "    prefill_seconds=prefill_seconds,native_request_seconds=prefill_seconds+elapsed,decode_seconds=elapsed")
ast.parse(text);(R/'worker1.py').write_text(text)
save('preparation1.json',dict(passed=True,previous_goal_turn_classification='progress: checkpoint7a88648b9 measured current verifier regions, rejected centered Q4 pilots and identified full-prompt draft admission failure on actual2K workloads.',base_worker_sha256=sha(S/'worker1.py'),worker_sha256=sha(R/'worker1.py'),worker_changes=changes,source_native_provenance_sha256=sha(N/'provenance1.json'),frozen_selected_serving_provenance_sha256=sha(SG/'provenance1.json'),target_prompt_tokens=len(request['token_ids']),target_rendered_prompt_sha256=hashlib.sha256(request['rendered_prompt'].encode()).hexdigest(),full_draft_prompt_tokens=len(full_ids),bounded_draft_prompt_tokens=len(bounded_ids),fixed_user_tail_tokens=192,original_task_suffix_preserved=True,input_user_messages_sha256=hashlib.sha256(json.dumps(body['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest(),no_native_model_or_precision_change=True,no_future_output_used_for_draft_prompt_construction=True,original_frozen_outputs_extracted_for_post_generation_check_only=True,model_execution=False,serving_speed_claim=False,goal_complete=False))
print(json.dumps(dict(prepared=True,target_tokens=len(request['token_ids']),full_draft_tokens=len(full_ids),bounded_draft_tokens=len(bounded_ids),task_suffix_preserved=True)))
