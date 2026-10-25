"""Exact full-corpus parity with the closed lifecycle parent; no timing claims."""
import hashlib

FIELDS=('batch_top1_parallel_enabled','batch_top1_parallel_max_rows','batch_top1_parallel_min_vocab',
 'batch_top1_legacy_host_dispatch_calls','batch_top1_parallel_host_dispatch_calls','batch_top1_counter_scope')

def compare(old,new,old_vectors,new_vectors,library_sha256,mode):
 assert mode in ('0','1')
 extra={'sampler_mode','batch_top1_dispatch_before','batch_top1_dispatch_after'}
 assert set(new)==set(old)|extra and new['sampler_mode']==mode
 assert new['garnet_dll_sha256']==library_sha256 and old['garnet_dll_sha256']!=library_sha256
 assert old['status']==new['status']=='bf16_padded_lifecycle_finished_needs_replay'
 for key,value in old.items():
  if key not in ('source','garnet_dll_sha256','external_compressed_pool'):assert new[key]==value,('Full lifecycle parity differs',key)
 assert len(new['cases'])==14 and [(c['group'],c['repeat']) for c in new['cases']]==[(g,r) for g in range(7) for r in range(2)]
 pool=new['external_compressed_pool'];prior=old['external_compressed_pool']
 assert set(pool)==set(prior)|set(FIELDS)
 for key,value in prior.items():
  if key!='device_used_bytes':assert pool[key]==value,('Owned model buffer accounting differs',key)
 assert type(pool['device_used_bytes']) is int and pool['device_used_bytes']>=0
 before,after=(new['batch_top1_dispatch_'+phase] for phase in ('before','after'))
 assert set(before)==set(after)==set(FIELDS) and before=={n:pool[n] for n in FIELDS}
 for snapshot in (before,after):
  assert type(snapshot[FIELDS[0]]) is bool and snapshot[FIELDS[0]]==(mode=='1')
  assert snapshot[FIELDS[1]]==128 and snapshot[FIELDS[2]]==8192
  assert snapshot[FIELDS[5]]=='host dispatch attempts including capture construction; excludes GPU graph replays'
  for key in FIELDS[3:5]:assert type(snapshot[key]) is int and snapshot[key]>=0
 selected,other=(FIELDS[4],FIELDS[3]) if mode=='1' else (FIELDS[3],FIELDS[4])
 assert after[selected]>before[selected] and after[other]==before[other]
 assert set(old_vectors)==set(new_vectors) and len(old_vectors)==102
 for key,data in old_vectors.items():assert len(data)==81920 and new_vectors[key]==data,('Raw full-model vector differs',key)
 return {'status':'complete_lifecycle_parent_parity_and_dispatch_passed','mode':mode,'cases':14,
  'raw_vectors_exact':102,'raw_vector_bytes':sum(len(v) for v in new_vectors.values()),
  'raw_vectors_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in new_vectors.items()},
  'dispatch_before':before,'dispatch_after':after,
  'scope':'All14old lifecycle reports, IDs, logical/physical state, ordinary/MTP/callback/prefix/reset/retirement metrics and102raw hidden/normalized vectors exact. DLL/source provenance and device-wide memory snapshot are separately recorded; owned allocation metadata is exact. Host dispatch calls include capture construction and exclude graph replays. No broad quality, actual4K, HTTP or speed qualification.'}
