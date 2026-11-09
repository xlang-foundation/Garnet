"""Copy the frozen wide transport auditor and explicitly bind retained-cache policy."""
from pathlib import Path
import hashlib,json,shutil
r=Path(__file__).resolve().parent;j=r.parent/'mistral-prefill-completion-wide1';dst=r/'audit-tools1'
assert not dst.exists();dst.mkdir()
for name in ['audit_half_transport1.py','audit_policies1.py']:shutil.copyfile(j/'audit-tools3'/name,dst/name)
p=dst/'audit_half_transport1.py';text=p.read_text();changes=[]
def edit(a,b):
    global text
    assert text.count(a)==1,(a[:100],text.count(a));text=text.replace(a,b);changes.append(dict(old=a,new=b))
edit("    require(profile.get('native_cached_chunks') is bulk_enabled and profile['prefix_entries']==profile['prefix_host_bytes']==0,'incorrect cold tracking/retention profile')", """    require(not bulk_enabled and profile.get('native_cached_chunks') is prefix_policy
            and profile['prefix_entries']==128 and profile['prefix_host_bytes']==2684354560
            and profile['prefix_capture_policy']=='prefill_only_v1'
            and profile['prefix_shared_blocks'] is True and profile['prefix_direct_capture'] is True,
            'incorrect bounded lifecycle retention policy')""")
edit("""            st=e['stats'];require(st['retention_enabled'] is False and all(st[k]==0 for k in ['entries','accounted_host_bytes','hits','dense_capture_copied_bytes','dense_capture_compared_bytes']),'retained or restored a cold prefix')""", """            st=e['stats'];require(prefix_policy and st['retention_enabled'] is True
                    and 0<=st['entries']<=profile['prefix_entries']
                    and 0<=st['accounted_host_bytes']<=profile['prefix_host_bytes']
                    and st['accounted_host_bytes']==st['private_host_bytes']+st['shared_resource_bytes']
                    and st['capture_policy']=='prefill_only_v1','invalid retained prefix accounting/policy')""")
p.write_text(text);compile(text,str(p),'exec')
source=(j/'audit_admission3.py').read_text();fused=source[source.index('def fused_check('):source.index('\nfiles={}')]
fused=fused.replace("batches=[e for e in events if e['event']=='native_batch']", "batches=[e for e in events if e['event']=='native_batch' and (e['stage']!='prefill' or any(q['prefill_tokens'] for q in e['requests']))]",1)
(dst/'audit_fused_route1.py').write_text('from audit_half_transport1 import require\n'+fused+'\n')
(r/'audit-preparation1.json').write_text(json.dumps(dict(passed=True,changes=changes,head_only_batches_excluded_from_fused_body_routes=True,original_sha256=hashlib.sha256((j/'audit-tools3/audit_half_transport1.py').read_bytes()).hexdigest(),local_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),no_native_or_model_changes=True),indent=2)+'\n')
print('Prepared retained-cache transport and fused-route audit helpers')
