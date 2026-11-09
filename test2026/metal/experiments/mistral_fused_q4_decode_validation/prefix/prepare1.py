"""Compose frozen prefix observers with the qualified fusion worker; no inference."""
from pathlib import Path
import ast,hashlib,json,shutil
P=Path(__file__).resolve().parent;M=P.parent;R=M/'mistral-fused-q4-decode-native1';L=M/'mistral-wide-half-prefix-http1'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):
    p=P/n;assert not p.exists(),n;p.write_text(json.dumps(d,indent=2)+'\n')
assert not (P/'preparation1.json').exists()
assert sha(R/'provenance1.json')=='a24b9df29897b392cf94556ecf38fee8eb112424e491c3e46fe2cad82f340c3c'
for n in ['model-audit2.json','verification-audit3.json','admission-audit1.json']:assert read(R/n)['passed']
for name,base in [('tools1',R/'tools1'),('audit-tools1',L/'audit-tools1')]:
    dest=P/name
    if dest.exists():
        # A previous setup copied read-only files and failed before any write.
        expected={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()}
        actual={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
        assert expected==actual,'partial setup modified input copies'
    else:shutil.copytree(base,dest)
    for p in [dest]+list(dest.rglob('*')):
        if not p.is_symlink():p.chmod(p.stat().st_mode|0o200|(0o100 if p.is_dir() else 0))
p=P/'tools1/local_coder_batch_worker.py';s=p.read_text();changes=read(L/'preparation1.json')['changes']
for e in changes:
    assert s.count(e['old'])==1,e['old'][:100];s=s.replace(e['old'],e['new'])
p.write_text(s);ast.parse(s)
shutil.copyfile(L/'fixture1.json',P/'fixture1.json')
profile=read(L/'profile-template1.json');profile.update(profile_id='mistral-fused-q4-decode-prefix-live1',fused_decode_rows4=True,scope='Untimed native fused FP32 Q4 decode prefix return/append/edit with three fully-prefilled survivors; no speed/quality qualification.')
profile['prefix_identity'][1]='mistral-fused-q4-decode-prefix-live1';profile['prefix_identity'][4]='native-fused-q4-decode-prefix-lifecycle1';save('profile-template1.json',profile)
b=read(L/'bindings1.json');b.update(wide_native_root=str(R),native_provenance_sha256=sha(R/'provenance1.json'),source=str(R/'source3'),source_bindings=str(R/'source-bindings3.json'),source_binding_sha256=sha(R/'source-bindings3.json'),library=str(R/'library3'),library_sha256=sha(R/'library3/libgarnet.dylib'),native_model_gate_sha256=sha(R/'model-audit2.json'),native_admission_gate_sha256=sha(R/'admission-audit1.json'),fixture_sha256=sha(P/'fixture1.json'),worker_sha256=sha(p));save('bindings1.json',b)
s=(L/'launch1.py').read_text().replace("J/'admission-audit3.json'","J/'admission-audit1.json'")
s=s.replace("for p in [ROOT/'bindings1.json'","bind(J/'verification-audit3.json');assert read(J/'verification-audit3.json')['passed']\nfor p in [ROOT/'bindings1.json'")
s=s.replace("+sorted((ROOT/'tools1').glob('*.py'))", "+sorted((ROOT/'tools1').glob('*.py'))+sorted((ROOT/'audit-tools1').glob('*.py'))+[ROOT/'audit_lifecycle1.py']")
(P/'launch1.py').write_text(s)
s=(L/'run_http1.py').read_text()
needle="    first_event(folder/'D/request-0.events.jsonl',jobs['D'])\n    state['transitions'].append(dict(trigger='D-first-token-all-three-seeds-prefilled',submitted='A',monotonic=time.monotonic()));save()"
replacement="""    began=time.monotonic();seed_acks=set()
    while seed_acks!={1,2,3}:
     if time.monotonic()-began>600:raise RuntimeError('Seed native prefill acknowledgment timeout')
     worker_log=folder/'server/worker-stdout.jsonl'
     if worker_log.exists():
      for line in worker_log.read_text().splitlines():
       try:event=json.loads(json.loads(line)['line'])
       except (ValueError,KeyError):continue
       if event.get('event')=='native_prefill_chunk' and event.get('final'):seed_acks.add(event['native_request'])
     if any(f.done() for f in jobs.values()):raise RuntimeError('A seed completed before newcomer admission')
     if seed_acks!={1,2,3}:time.sleep(.01)
    state['transitions'].append(dict(trigger='native-final-prefill-acks-1-2-3',seed_native_requests=sorted(seed_acks),submitted='A',monotonic=time.monotonic()));save()"""
assert s.count(needle)==1;s=s.replace(needle,replacement);(P/'run_http1.py').write_text(s)
s=(R/'audit_admission1.py').read_text();function=s[s.index('def decode_check('):s.index('\ndef fused_check(')]
needle="        if event['event']=='native_batch':\n            require(pending is None,'missing native decode observation')\n            pending=event"
replacement="""        if event['event']=='native_batch':
            require(pending is None,'missing native decode observation')
            if event['stage']=='prefill' and not any(q['prefill_tokens'] for q in event['requests']):continue
            pending=event"""
assert function.count(needle)==1;function=function.replace(needle,replacement)
(P/'audit-tools1/audit_decode_route1.py').write_text('from audit_half_transport1 import require\n'+function+'\n')
s=(L/'audit_lifecycle2.py').read_text().replace("ROOT/'lifecycle-audit2.json'","ROOT/'lifecycle-audit1.json'")
s=s.replace("from audit_fused_route1 import fused_check","from audit_fused_route1 import fused_check\nfrom audit_decode_route1 import decode_check")
s=s.replace("policy=check_policy(es);fused=fused_check(es,profile)","policy=check_policy(es);fused=fused_check(es,profile);decode=decode_check(es,profile)\n    require(state['transitions'][0]['trigger']=='native-final-prefill-acks-1-2-3' and state['transitions'][0]['seed_native_requests']==[1,2,3],'seed readiness was not observed at native boundary')")
s=s.replace("transport=transport,policy=policy,fused=fused,","transport=transport,policy=policy,fused=fused,decode=decode,")
s=s.replace("sha(ROOT/'audit_lifecycle2.py')","sha(ROOT/'audit_lifecycle1.py')")
old="[ROOT/'prepare_audit1.py',ROOT/'audit-preparation1.json',ROOT/'audit_lifecycle1.py',ROOT/'audit_lifecycle2.py',ROOT/'collection-clarification1.json']"
assert old in s;s=s.replace(old,"[ROOT/'prepare1.py',ROOT/'preparation1.json',ROOT/'audit_lifecycle1.py']")
needle="        'wrong_fused_route':lambda d:event(d,'native_fused_mlp').update(blocks=0),"
assert needle in s
s=s.replace(needle,needle+"\n        'missing_decode_fusion':lambda d:next(e for e in d['cached']['events'] if e['event']=='native_fused_q4_decode' and e['blocks']==32).update(blocks=0),\n        'fused_partial_or_prefill':lambda d:next(e for e in d['cached']['events'] if e['event']=='native_fused_q4_decode' and e['blocks']==0).update(blocks=32),\n        'wrong_decode_copy':lambda d:event(d,'native_fused_q4_decode').update(input_copies=33),\n        'wrong_decode_preparation':lambda d:event(d,'native_fused_q4_decode_prepared')['sites'].__setitem__(0,0),")
(P/'audit_lifecycle1.py').write_text(s)
save('preparation1.json',dict(passed=True,previous_goal_turn_classification='progress: checkpoint12d787a7b integrated native fusion and completed diagnostic serving.',base_native_provenance_sha256=sha(R/'provenance1.json'),base_prefix_provenance_sha256=sha(L/'provenance1.json'),base_worker_sha256=sha(R/'tools1/local_coder_batch_worker.py'),worker_sha256=sha(p),observer_changes=changes,native_plans_unchanged=True,no_new_build=True,head_only_full_hit_excluded_from_body_observations=True,seed_readiness='Wait for all three native final-prefill acknowledgments before A submission; audit actual newcomer plan ordering.',fixture_unchanged_sha256=sha(L/'fixture1.json'),initial_setup_failure='Copied files retained frozen read-only mode; first attempted worker write raised PermissionError before any mutation. Fresh copies verified byte-identical and only destination copies made writable.',performance_measurement=False,goal_complete=False))
for f in P.rglob('*.py'):ast.parse(f.read_text())
print(json.dumps(dict(prepared=True,root=str(P),worker_sha256=sha(p),fixture_sha256=sha(P/'fixture1.json'))))
