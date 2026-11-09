"""Original frozen C4 user inputs; helper sees a fixed user tail, no future output."""
from pathlib import Path
import hashlib,json
from tokenizers import Tokenizer
from jinja2.sandbox import SandboxedEnvironment
R=Path(__file__).resolve().parent;M=R.parent
read=lambda p:json.loads(p.read_text())
sg=M/'mistral-fused-mlp-serving1';b=read(M/'mistral-bounded-draft-long1/input1.json')
plan=next(x for x in read(sg/'serving-manifest1.json')['plans'] if x['id']=='p2048-c4-o128')
bodies=read(Path(plan['cases']));requests=read(Path(plan['workload']))['requests'];assert len(bodies)==len(requests)==4
folder=Path(b['draft_model']);tok=Tokenizer.from_file(str(folder/'tokenizer.json'));template=SandboxedEnvironment().from_string(read(folder/'tokenizer_config.json')['chat_template'])
jobs=[]
for label,body,request in zip('ABCD',bodies,requests):
 assert len(body['messages'])==1 and body['messages'][0]['role']=='user'
 text=body['messages'][0]['content'];ids=tok.encode(text,add_special_tokens=False).ids
 tail=tok.decode(ids[-192:],skip_special_tokens=False);assert text[text.rindex('Task:'):] in tail
 prompt=template.render(messages=[dict(role='user',content=tail)],add_generation_prompt=True,enable_thinking=False,tools=None)
 assert len(tok.encode(prompt,add_special_tokens=False).ids)+8<=512
 jobs.append(dict(label=label,prompt_ids=request['token_ids'],rendered=request['rendered_prompt'],draft_prompt=prompt,limit=32))
inputs=dict(scope=__doc__,target_model=b['target_model'],target_tokenizer_sha256=read(M/'mistral-gqa16-value-native1/input2.json')['target_tokenizer_sha256'],draft_model=b['draft_model'],draft_revision=b['draft_revision'],draft_tokenizer_sha256=b['draft_tokenizer_sha256'],stops=b['target_stops'],draft_stops=b['draft_stops'],capacity=4096,batch=4,draft_capacity=512,draft_token_limit=8,verification_tokens=4,jobs=jobs,reference_output_access_during_generation=False,correctness_only=True,performance_claim=False,no_comparator_execution=True,goal_complete=False)
p=R/'productive-input1.json';assert not p.exists();p.write_text(json.dumps(inputs,indent=2)+'\n')
print('Prepared original C4 prompts',[(j['label'],len(j['prompt_ids']),j['limit']) for j in jobs])
