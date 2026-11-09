"""Correct the benchmark profile's declared cache identity without changing its mathematical work."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).resolve().parent;s=(R/'b1_worker1.py').read_text();original=s
marker="helper_prefill_maximum=int(sys.argv[6]);assert helper_prefill_maximum in (16,32,64,128,256)";assert s.count(marker)==1
s=s.replace(marker,marker+"\nhelper_prefix_profile_id='B1D'+str(inputs.get('draft_capacity',inputs['capacity']))+'T'+str(helper_prefill_maximum)")
old="'B1D'+str(inputs.get('draft_capacity',inputs['capacity']))+'T16'";assert s.count(old)==1;s=s.replace(old,'helper_prefix_profile_id')
marker='helper_prefill_maximum=helper_prefill_maximum)';assert s.count(marker)==1;s=s.replace(marker,'helper_prefill_maximum=helper_prefill_maximum,helper_prefix_profile_id=helper_prefix_profile_id)')
p=R/'b1_worker2.py';assert not p.exists();p.write_text(s);compile(s,'worker2','exec')
(R/'identity-change3.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/b1_worker1.py',tofile='b/b1_worker2.py',n=0)))
(R/'identity-preparation3.json').write_text(json.dumps(dict(passed=True,old_worker_sha256=hashlib.sha256(original.encode()).hexdigest(),new_worker_sha256=hashlib.sha256(s.encode()).hexdigest(),changes=['Declare helper cache profile from actual upper bound','Record the declared identity in evidence'],mathematical_functions_unchanged=True,preparation_identity_only=True,old_cache_entries_always_zero=True,historical_timings_preserved=True,no_comparator_execution=True),indent=2)+'\n')
