"""Bind GPU Q/KV packing to the independently qualified revision2 payloads."""
import sys
from pathlib import Path
import hashlib,json
p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
load=lambda q:json.loads(q.read_text())
def sha(q):
    h=hashlib.sha256()
    with q.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
a=load(p/'numeric-audit2.json');assert a['passed'] and a['qualified_variants']==[True,True]
b=load(p/'source-bindings3.json')
for name,h in b['sha256'].items():assert sha(p/name)==h
before=load(p/'payload-hashes2.json');payloads={};metas={};float_pairs=0;half_pairs=0
assert b['sha256']['numeric-policy-budget1.json']==a['numeric_budget_sha256']
for kind in ['synthetic','trained']:
    prior=p/f'raw-{kind}2';current=p/f'raw-{kind}3'
    old=load(prior/'correctness1.json');r=load(current/'correctness1.json')
    assert load(p/(current.name+'-status.json'))['returncode']==0
    assert r['collected'] and r['cases']==old['cases']
    assert r['packed_half_storage'] and r['relaxed_precision'] and r['Q_pack_inside_gpu_span'] and r['KV_pack_inside_gpu_span']
    assert not r['half_KV_fixture_prepared_before_timing'] and r['half_scratch_poisoned_before_each_candidate']
    assert r['register_layout_status']==[0]*32 and r['half_policy_layout_capacities']==old['half_policy_layout_capacities']
    assert r['half_policy_coordinate_parent_words']==old['half_policy_coordinate_parent_words']
    actual={}
    for name,h in before[prior.name].items():
        assert sha(prior/name)==sha(current/name)==h,name
        actual[name]=h
        if name.endswith('.f16'):half_pairs+=(current/name).stat().st_size//2
        else:float_pairs+=(current/name).stat().st_size//4
    payloads[current.name]=actual;metas[current.name]=sha(current/'correctness1.json')
result=dict(passed=True,qualified_variants=[True,True],scope=__doc__,numeric_reference_audit_sha256=sha(p/'numeric-audit2.json'),numeric_budget_sha256=a['numeric_budget_sha256'],source_bindings_sha256=sha(p/'source-bindings3.json'),metadata_sha256=metas,exact_float_parent_word_pairs=float_pairs,exact_half_parent_word_pairs=half_pairs,all_staging_parents_poisoned_before_each_candidate=True,Q_and_KV_pack_inside_gpu_span=True,original_FP32_contract_qualified=False,native_integration=False,quality_qualified=False,serving_speed_claim=False,goal_complete=False)
(p/'payload-hashes3.json').write_text(json.dumps(payloads,indent=2)+'\n')
(p/'numeric-audit3.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
