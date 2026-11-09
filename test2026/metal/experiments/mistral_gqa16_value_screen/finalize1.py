"""Retain the selected value policy and bind the complete operator evidence."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(name,value):
 p=R/name;assert not p.exists(),name;p.write_text(json.dumps(value,indent=2)+'\n')
a=json.loads((R/'audit1.json').read_text());assert a['passed']
selected=[m for m in a['timing'] if m['method']==('tile64' if m['batch']==1 else 'tile32')]
assert len(selected)==24 and all(m['paired_ratio_vs_current']>1 and m['paired_ratio_vs_plain_vector4']>1 for m in selected)
save('decision1.json',dict(passed=True,decision='Advance shared-value GQA16 to isolated native integration; retain current runtime until qualification.',selected_tile_by_batch={'1':64,'4':32},native_visible_threshold=512,fixed_profile=dict(batch=[1,4],chunk=4,heads=32,kv_heads=8,dim=128,capacity=4096,gated=False),selection_evidence=selected,accumulation='Ascending keys, same per-component FP32 chain; no parallel sum.',new_global_gpu_workspace_bytes=0,dynamic_threadgroup_bytes_by_batch={'1':12288,'4':6144},native_integration_qualified=False,serving_speed_claim=False,quality_qualified=False,goal_complete=False,no_comparator_execution=True,previous_goal_turn='progress: merged origin/main and passed configured native/tokenizer checks',next=['Reconcile latest native QK source with merged upstream main in an isolated source tree.','Add opt-in prepared value PSOs, bounded route, distinct dispatch counter and ownership tests.','Qualify ON/default-OFF native and full-model outputs before serial paired measurement.']))
files={str(p.relative_to(R)):sha(p) for p in sorted(R.rglob('*')) if p.is_file()}
save('provenance1.json',dict(passed=True,scope='Complete pilot and final shared-value operator evidence, including immutable native-source bindings.',files_sha256=files,audit_sha256=sha(R/'audit1.json'),no_gpu_execution_at_finalize=True,no_comparator_execution=True,goal_complete=False))
print('Bound',len(files),'files; provenance',sha(R/'provenance1.json'))
