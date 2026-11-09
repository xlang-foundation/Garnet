"""Run the same return/edit sequence with an80MiB host-prefix budget."""
import sys
from pathlib import Path
import hashlib,json,shutil
root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
base=root.parent/'mistral-half-prefix-http1'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert load(base/'half-http-audit1.json')['passed'] and load(base/'frozen-fp32-output-audit2.json')['passed']
tools=root/'tools1';assert not tools.exists();tools.mkdir()
for p in sorted((base/'tools1').iterdir()):
    assert p.is_file();shutil.copy2(p,tools/p.name)
for name in ['fixture2.json','run_http1.py','launch1.py']:
    shutil.copy2(base/name,root/name)
runner=root/'run_http1.py';text=runner.read_text();assert text.count("[('uncached',False),('cached',True)]")==1
text=text.replace("[('uncached',False),('cached',True)]","[('pressure',True)]");runner.chmod(0o644);runner.write_text(text)
profile=load(base/'profile-template1.json');profile.update(profile_id='mistral-half-prefill-prefix-pressure1',prefix_host_bytes=80*1024*1024,
    scope='Untimed same nine-request native half-prefill return/edit protocol under80MiB host-prefix pressure. One fresh pressure engine, compared with saved uncached half-policy control; no timing/quality admission.')
(root/'profile-template1.json').write_text(json.dumps(profile,indent=2)+'\n')
b=load(base/'bindings1.json');b.update(scope=profile['scope'],baseline_half_http_audit_sha256=sha(base/'half-http-audit1.json'),
    baseline_uncached_worker_stdout_sha256=sha(base/'uncached/server/worker-stdout.jsonl'))
(root/'bindings1.json').write_text(json.dumps(b,indent=2)+'\n')
# Keep the existing protocol invariants, adapting only pressure-dependent hit
# expectations. Full outputs/target states are checked separately against the
# already-collected uncached engine, without rerunning it.
text=(base/'audit_http2.py').read_text()
changes=[("cached=label=='cached'","cached=label=='pressure'"),
 ("  expected=({'A-return':211,'append':128,'changed-tail':128}.get(name,0) if cached else 0)","""  expected=c['cached_tokens'] if cached else 0
  allowed={'A-return':{0,128,211},'append':{0,128},'changed-tail':{0,128}}.get(name,{0})
  assert expected in allowed,(name,expected,allowed)"""),
 ("   if name=='A-return':assert first['prefill_tokens']==0", "   if name=='A-return':assert (first['prefill_tokens']==0)==(expected==len(entry['prompt_ids']))"),
 ("s['accounted_host_bytes']<=384*1024*1024", "s['accounted_host_bytes']<=80*1024*1024"),
 ("  assert stats[-1]['prefill_capture_skipped_mixed']==9 and stats[-1]['hits']==3 and stats[-1]['misses']==6", """  hits=sum(c['cached_tokens']>0 for c in byname.values())
  assert stats[-1]['prefill_capture_skipped_mixed']==9 and stats[-1]['hits']==hits and stats[-1]['misses']==9-hits
  assert hits>=1 and stats[-1]['evictions']>0""")]
for a,c in changes:assert text.count(a)==1,(a,text.count(a));text=text.replace(a,c)
compile(text,str(root/'audit_pressure_protocol1.py'),'exec');(root/'audit_pressure_protocol1.py').write_text(text)
print(json.dumps(dict(prepared=True,host_prefix_budget=80*1024*1024,original_protocol_fixture_unchanged=sha(root/'fixture2.json')==sha(base/'fixture2.json'))))
