"""CPU-only draft input admission; never open target output/control files."""
from pathlib import Path
import hashlib,json,sys
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
R=Path(__file__).resolve().parent;M=R.parent;S=M/'mistral-fused-q4-decode-coder1';SG=M/'mistral-fused-mlp-serving1'
DEST=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else R/'draft-long-context-admission1.json'
assert not DEST.exists()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs=read(S/'input1.json');model=Path(inputs['draft_model']);tc=read(model/'tokenizer_config.json');template=SandboxedEnvironment().from_string(tc['chat_template']);tok=Tokenizer.from_file(str(model/'tokenizer.json'));rows=[]
for case in ['p2048-c1-o128','p2048-c4-o128']:
    bodies=read(SG/'cases1'/(case+'.json'))
    for i,body in enumerate(bodies):
        messages=body['messages'];prompt=template.render(messages=messages,add_generation_prompt=True,enable_thinking=False,tools=None);ids=tok.encode(prompt,add_special_tokens=False).ids
        assert len(messages)==1 and messages[0]['role']=='user' and isinstance(messages[0]['content'],str)
        content_ids=tok.encode(messages[0]['content'],add_special_tokens=False).ids
        tail=tok.decode(content_ids[-192:],skip_special_tokens=False)
        bounded=template.render(messages=[dict(role='user',content=tail)],add_generation_prompt=True,enable_thinking=False,tools=None);bid=tok.encode(bounded,add_special_tokens=False).ids
        rows.append(dict(case=case,request=i,full_draft_prompt_tokens=len(ids),full_prompt_admitted_by_existing_profile=len(ids)+inputs['draft_token_limit']<=inputs['draft_capacity'],bounded_draft_prompt_tokens=len(bid),bounded_initial_admission=len(bid)+inputs['draft_token_limit']<=inputs['draft_capacity'],draft_capacity=inputs['draft_capacity'],maximum_draft_tokens=inputs['draft_token_limit'],target_output_limit=body['max_tokens'],original_messages_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),draft_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),bounded_draft_prompt_sha256=hashlib.sha256(bounded.encode()).hexdigest()))
report=dict(scope=__doc__,passed=True,rows=rows,existing_full_prompt_initially_admitted=sum(x['full_prompt_admitted_by_existing_profile'] for x in rows),bounded_initially_admitted=sum(x['bounded_initial_admission'] for x in rows),proposed_next_experiment='Keep the complete target prompt/context unchanged. Use a fixed192-token tail of draft user content with intact draft chat framing, append only the already-generated target prefix, retain resident reconciliation and the512-token guard. Exact target state and matched2K C1 serving required before speed claim.',bounded_draft_policy_measured=False,no_target_context_change=True,no_model_execution=True,no_target_future_outputs_opened=True,model_tokenizer_sha256=sha(model/'tokenizer.json'),files_sha256={str(p):sha(p) for p in [S/'input1.json',model/'tokenizer.json',model/'tokenizer_config.json',SG/'cases1/p2048-c1-o128.json',SG/'cases1/p2048-c4-o128.json']},goal_complete=False)
DEST.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['files_sha256','rows']},indent=2));print(json.dumps(rows,indent=2))
