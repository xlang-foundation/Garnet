"""Extend the qualified helper-bound screen to32/64/128 without changing captured runs."""
from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'run_qualification1.py').read_text();s=s.replace("for cap in [16,256]:","for cap in [32,64,128]:").replace("'qualification1-preflight.json'","'sweep2-preflight.json'").replace("'qualification1-postflight.json'","'sweep2-postflight.json'")
s=s.replace("['run_qualification1.py','b1_worker1.py'","['run_sweep2.py','qualification-audit1.json','b1_worker1.py'")
(R/'run_sweep2.py').write_text(s)
s=(R/'audit_qualification1.py').read_text();s=s.replace("need(read(R/'qualification1-postflight.json')['passed'],'terminal collection gate');pre=read(R/'qualification1-preflight.json');", "need(read(R/'qualification1-postflight.json')['passed'] and read(R/'sweep2-postflight.json')['passed'],'terminal collection gates');pre=read(R/'sweep2-preflight.json');")
s=s.replace('for cap in [16,256]:','for cap in [16,256,32,64,128]:')
s=s.replace("full_target_output_ids=512,full_target_state_digest_pairs=260","full_target_output_ids=1280,full_target_state_digest_pairs=650")
s=s.replace("'cap16-result1.json','cap256-result1.json'","'cap16-result1.json','cap256-result1.json','cap32-result1.json','cap64-result1.json','cap128-result1.json','sweep2-preflight.json','sweep2-postflight.json'")
s=s.replace("R/'qualification-audit1.json'","R/'qualification-audit2.json'")
(R/'audit_qualification2.py').write_text(s)
for n in ['run_sweep2.py','audit_qualification2.py']:compile((R/n).read_text(),n,'exec')
