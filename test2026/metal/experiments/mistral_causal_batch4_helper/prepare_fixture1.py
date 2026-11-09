"""Offline committed-prefix fixtures for native B4 helper ownership qualification."""
from pathlib import Path
import hashlib,json
from jinja2.sandbox import SandboxedEnvironment
from tokenizers import Tokenizer
R=Path(__file__).resolve().parent;M=R.parent
b=json.loads((M/'mistral-bounded-draft-long1/input1.json').read_text())
folder=Path(b['draft_model']);cfg=json.loads((folder/'tokenizer_config.json').read_text());template=SandboxedEnvironment().from_string(cfg['chat_template']);tok=Tokenizer.from_file(str(folder/'tokenizer.json'))
users={'A':'Write a Python function add(a, b) that returns their sum. Return code only.','B':'Write JavaScript clamp(x, low, high). Return code only.','C':'Write C++ int square(int x). Return code only.','D':'Write SQL selecting active users ordered by name. Return SQL only.','E':'Write a Python function is_even(n). Return code only.'}
base={key:template.render(messages=[dict(role='user',content=value)],add_generation_prompt=True,enable_thinking=False,tools=None) for key,value in users.items()}
frames=[
 {'queries':[('A','',8),('B','',3),('C','',5),('D','',1)]},
 {'queries':[('A','def add(a, b):\n',3),('C','int square(int x) {\n',6)]},
 {'release':'B','claim':'E','queries':[('E','def is_even(n):\n    ',8),('D','SELECT name FROM users',2)]},
 {'queries':[('A','def add(a, b):\n    return a + b\n',4),('E','def is_even(n):\n    return n % 2 == 0',1),('C','int square(int x) { return x * x; }',6),('D','SELECT id, name FROM users WHERE active = 1',8)]},
 {'queries':[('A','def add(a, b):\n    return a + b\n',4)]}
]
for frame in frames:
 frame['queries']=[dict(label=label,committed_text=prefix,prompt=base[label]+prefix,prompt_ids=tok.encode(base[label]+prefix,add_special_tokens=False).ids,draft_limit=limit) for label,prefix,limit in frame['queries']]
 for q in frame['queries']:assert len(q['prompt_ids'])+q['draft_limit']<=512
fixture=dict(scope=__doc__,draft_model=b['draft_model'],draft_revision=b['draft_revision'],draft_tokenizer_sha256=b['draft_tokenizer_sha256'],capacity=512,batch=4,prefill_chunk=16,initial=['A','B','C','D'],frames=frames,draft_stops=b['draft_stops'],fixture_prefixes_are_explicit_committed_inputs=True,target_generated=False,performance_claim=False,no_comparator_execution=True,goal_complete=False)
p=R/'input1.json';assert not p.exists();p.write_text(json.dumps(fixture,indent=2)+'\n')
print('Prepared',sum(len(f['queries']) for f in frames),'helper contexts across',len(frames),'frames.')
