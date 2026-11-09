"""Pinned text-prefix lineage fixture, including a small-tail batch-layout change."""
from pathlib import Path
import json
R=Path(__file__).resolve().parent;M=R.parent;B=M/'mistral-wide-lifecycle1'
base=json.loads((B/'input2.json').read_text());prompt=base['jobs'][0]['prompt_ids'];assert len(prompt)==2048
jobs=[]
for item in base['jobs'][:4]:jobs.append(dict(item,limit=4 if item['label']=='A' else 64))
append=[1117,1032,22134,1676]
late=list(prompt);late[1799]=(late[1799]+1)%32768
early=list(prompt);early[511]=(early[511]+1)%32768
followups=[dict(label='A_exact',prompt_ids=prompt,limit=4),dict(label='B_exact',prompt_ids=jobs[1]['prompt_ids'],limit=4),dict(label='A_append',prompt_ids=prompt+append,limit=4),dict(label='A_suffix',prompt_ids=late,limit=4),dict(label='A_early',prompt_ids=early,limit=4)]
twins=[dict(label='twin_'+str(i),prompt_ids=prompt+append,limit=4) for i in [0,1]]
inputs=dict(target_model=base['target_model'],target_tokenizer_sha256=base['target_tokenizer_sha256'],stops=[],capacity=4096,batch=4,prefill_up_to=512,verification_tokens=4,host_entries=64,host_bytes=2*1024**3,profile_identity='scheduled-B4-C4096-prefill-up512-verify4-fp32-v1',jobs=jobs,followups=followups,twins=twins,reference_path=str(R/'ordinary-off-result1.json'),scope='Untimed retained-prefix correctness. Empty stops keep survivor budgets deterministic; zero proposals are not a learned draft or speed benchmark.')
(R/'input1.json').write_text(json.dumps(inputs,indent=2)+'\n')
(R/'plan1.json').write_text(json.dumps(dict(scope='Native C++ token-prefix lineage and batch-layout reuse on the wide prepared profile.',source=str(B/'source1'),library=str(B/'build-on1/libgarnet.dylib'),arms=['ordinary-off','ordinary-cache','verify-cache','reverse-cache'],owner_checkpoints=['prefill','before_terminal'],survivor_admissions=[f['label'] for f in followups],survivor_rows_required=['B','C','D'],required_full_hits={'A_exact':2048,'B_exact':2051},required_partial_hits={'A_append':2048,'A_suffix':1792,'A_early':384},twin_reuse_minimum=2048,twin_full_hit_if_retained=2052,twin_reason='A_append first uses one active prefill row with a full four-token tail; the two twins later use two rows. Verify semantic prefix identity does not reuse numerically incompatible states across those layouts.',expected_output_ids=224,full_state_digest_equality=True,host_entries=64,host_bytes=2*1024**3,no_comparator_execution=True,performance_measured=False,goal_complete=False),indent=2)+'\n')
assert sum(j['limit'] for j in jobs+followups+twins)==224
print('Prepared exact/append/late/early/twin prefix fixture')
