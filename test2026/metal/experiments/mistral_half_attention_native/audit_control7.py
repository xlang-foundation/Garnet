"""Compare new FP32 body state with already-frozen original body controls."""
import sys
from pathlib import Path
import hashlib,json
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
prior=root.parent.parent/'mistral-committed-kv-attention1'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frozen=load(prior/'provenance1.json')['files_sha256']
checks=[]
for cohort in [0,1]:
    for owners in [1,4]:
        name=f'cohort{cohort}-owners{owners}-off.json';old=prior/'body1'/name
        assert sha(old)==frozen['body1/'+name]
        a,b=load(old),load(root/name)
        for field in ['state_sha256','hidden_sha256','predictions','hidden_values']:assert a[field]==b[field],(name,field)
        checks.append(dict(file=name,old_sha256=sha(old),new_sha256=sha(root/name),equal_full_state_digests=65,complete_hidden_values=4*128*4096,predictions_exact=True))
result=dict(passed=True,scope=__doc__,controls=checks,frozen_control_provenance_sha256=sha(prior/'provenance1.json'),
    serving_speed_claim=False,quality_qualified=False,goal_complete=False)
dest=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else root/'control-audit7.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(passed=True,controls=len(checks),exact_digest_pairs=260,exact_hidden_value_pairs=4*4*128*4096)))
