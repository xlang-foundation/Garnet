"""Retain prior ownership gates and extend their asserted span/route contracts."""
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent/'mistral-compact-verification-coexist1'
expected="{'mixed_unscheduled_row','wrong_known_mixed','wrong_slots_shape','wrong_native_epoch','negative_native_budget','submitted_over_budget','stale_tick_replay'}"
for source,target in [('audit_core1.py','audit_owner1.py'),('audit_probe1.py','audit_probe_core1.py')]:
 s=(B/source).read_text().split('def main():',1)[0]
 s=s.replace('C4/T1','C4/T4').replace('native B4/T1 route differs','native B4/T4 route differs')
 s=s.replace("metrics['commit_kv_restores']==0","metrics['commit_kv_restores']==(64 if any(q['submitted']>1 for q in q) else 0) and metrics['target_indexed_decode_dispatches']==225 and metrics['target_mpp_affine_dispatches']==0 and metrics['target_indexed_prefill_dispatches']==0")
 s=s.replace("m['commit_kv_restores']==0","m['commit_kv_restores']==(64 if any(q['submitted']>1 for q in native['requests']) else 0) and m['target_indexed_decode_dispatches']==225 and m['target_mpp_affine_dispatches']==0 and m['target_indexed_prefill_dispatches']==0")
 s=s.replace("actual['submitted']==actual['consumed']==1","actual['consumed']==1 and actual['submitted']==min(4,actual['remaining_before'])")
 s=s.replace("q['submitted']==q['consumed']==1","q['consumed']==1 and q['submitted']==min(4,q['remaining_before'])")
 s=s.replace("{'before_invalid_mixed','after_invalid_mixed'}","{'before_invalid_mixed','after_invalid_mixed','before_invalid_budget','after_invalid_budget'}")
 s=s.replace("{'mixed_unscheduled_row','wrong_known_mixed','stale_tick_replay'}",expected)
 s=s.replace('rejected_call_state_digest_pairs=260','rejected_call_state_digest_pairs=520')
 if source=='audit_core1.py':
  needle=" before=copy.deepcopy(bw['before_invalid_mixed']);after=copy.deepcopy(bw['after_invalid_mixed']);before.pop('label');after.pop('label');need(before==after,'rejected mixed call changed full state')"
 else:
  needle=" one=copy.deepcopy(bw['before_invalid_mixed']);two=copy.deepcopy(bw['after_invalid_mixed']);one.pop('label');two.pop('label');need(one==two,'invalid mixed call altered state')"
 assert s.count(needle)==1
 s=s.replace(needle,needle+"\n one=copy.deepcopy(bw['before_invalid_budget']);two=copy.deepcopy(bw['after_invalid_budget']);one.pop('label');two.pop('label');need(one==two,'invalid native budget changed full state')")
 (R/target).write_text(s)
print('Prepared strict wide ownership and probe auditors')
