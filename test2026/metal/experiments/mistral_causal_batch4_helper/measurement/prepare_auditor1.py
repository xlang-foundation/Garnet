"""Derive a trace-only audit; explicitly remove unavailable state-hash assertions."""
from pathlib import Path
import hashlib,json,difflib
R=Path(__file__).resolve().parent;Q=R.parent/'mistral-causal-batch4-helper1'
base=(Q/'audit_productive1.py').read_text();s=base[:base.index('def main():')];changes=[]
def edit(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b);changes.append(dict(old=a,new=b))
edit('"""Independent CPU audit of causal B4 drafts and ordinary target state equivalence."""','"""CPU causal trace audit for timed runs: no full-state equivalence admission."""')
edit("R=Path(__file__).resolve().parent", "R=Path(__file__).resolve().parent\nQ=R.parent/'mistral-causal-batch4-helper1'")
edit("str(R/'build-on1/libgarnet.dylib')", "str(Q/'build-on1/libgarnet.dylib')")
start=s.index("   snap=d['snapshots'][si]");end=s.index(" need(all(j['generated']==hist",start)
edit(s[start:end]," need(len(finished)==4,'target completion')\n")
edit('def audit(ordinary,candidate,inputs,ttok,dtok):','def audit(ordinary,candidate,inputs,ttok,dtok,reconcile_offset=0):')
edit('reconcile_count=0;survivor_pairs=0;', 'reconcile_count=reconcile_offset;survivor_pairs=0;')
start=s.index("  for phase in ('before','after_reconcile','after'):checkstates");end=s.index('  selected=[];',start)
edit(s[start:end],"  need(all(round[phase] is None for phase in ('before','after_reconcile','after')),'timed state capture policy')\n")
start=s.index("  touched={'ABCD'.index(x)");end=s.index("  need(round['call_first']",start)
edit(s[start:end],'')
edit('helper_reconciliation_calls=reconcile_count,','helper_reconciliation_calls=reconcile_count-reconcile_offset,')
edit('performance_claim=False,http_qualified=False,','performance_claim=False,full_state_capture_in_timed_runs=False,http_qualified=False,')
compile(s,str(R/'audit_trace1.py'),'exec');p=R/'audit_trace1.py';assert not p.exists();p.write_text(s)
(R/'trace-audit-adaptation1.patch').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='a/audit_productive1.py',tofile='b/audit_trace1.py')))
(R/'trace-audit-preparation1.json').write_text(json.dumps(dict(passed=True,base_path=str(Q/'audit_productive1.py'),base_sha256=hashlib.sha256(base.encode()).hexdigest(),trace_audit_sha256=hashlib.sha256(s.encode()).hexdigest(),changes=changes,full_state_equivalence_claim=False,raw_results_unchanged=True),indent=2)+'\n')
print('Prepared explicit trace-only audit adapter.')
